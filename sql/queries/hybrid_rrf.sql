-- ===========================================================================
-- Hybrid retrieval, reciprocal rank fusion.
--
-- Identical candidate generation to hybrid_linear.sql -- same two arms, same
-- index-preserving shape. Only the fusion differs, and the difference is the
-- point:
--
--   linear  needs both arms on a comparable scale, so it needs min-max
--           normalisation, so its results depend on the candidate set
--   RRF     never looks at the scores at all, only the positions
--
-- Because it ignores magnitudes, RRF cannot be fooled by the cosine-vs-ts_rank
-- scale mismatch that makes naive linear weighting behave like pure vector
-- search. It has no tuning beyond `rrf_k`, and it is what most production
-- hybrid systems actually use.
--
-- Parameters (psycopg named):
--   qvec, qtext, strategy, model, fanout, k, rrf_k, w_vec, w_txt
-- ===========================================================================

WITH vec_candidates AS (
    SELECT e.chunk_id,
           e.vec <=> %(qvec)s::vector AS distance
    FROM   embeddings_384 e
    WHERE  e.strategy = %(strategy)s
      AND  e.model    = %(model)s
    ORDER  BY e.vec <=> %(qvec)s::vector
    LIMIT  %(fanout)s
),
vec AS (
    SELECT chunk_id,
           1.0 - distance                        AS vec_score,
           ROW_NUMBER() OVER (ORDER BY distance) AS vec_rank
    FROM   vec_candidates
),

txt_candidates AS (
    SELECT c.id AS chunk_id,
           ts_rank_cd(c.tsv, websearch_to_tsquery('english', %(qtext)s), 32) AS ts_score
    FROM   chunks c
    WHERE  c.strategy = %(strategy)s
      AND  c.tsv @@ websearch_to_tsquery('english', %(qtext)s)
    ORDER  BY ts_score DESC
    LIMIT  %(fanout)s
),
txt AS (
    SELECT chunk_id,
           ts_score,
           ROW_NUMBER() OVER (ORDER BY ts_score DESC) AS txt_rank
    FROM   txt_candidates
),

fused AS (
    SELECT chunk_id, v.vec_score, t.ts_score, v.vec_rank, t.txt_rank
    FROM   vec v FULL OUTER JOIN txt t USING (chunk_id)
)

SELECT c.id AS chunk_id,
       c.story,
       c.section,
       c.body,
       c.parent_body,
       f.vec_score,
       f.ts_score AS txt_score,
       f.vec_rank,
       f.txt_rank,
       -- Reciprocal rank fusion (Cormack et al., 2009).
       --
       -- rrf_k = 60 is the constant from the paper and the one everyone uses.
       -- It damps the head of each list: smaller values trust rank 1 more,
       -- larger values flatten both lists toward a round-robin merge.
       --
       -- A document missing from an arm contributes exactly 0 -- that is
       -- canonical RRF. COALESCE is what implements "missing", since the
       -- FULL OUTER JOIN leaves the absent side's rank NULL.
       COALESCE(%(w_vec)s / (%(rrf_k)s + f.vec_rank), 0.0) +
       COALESCE(%(w_txt)s / (%(rrf_k)s + f.txt_rank), 0.0) AS score
FROM   fused f
JOIN   chunks c ON c.id = f.chunk_id
ORDER  BY score DESC, f.chunk_id
LIMIT  %(k)s;

-- A variant worth knowing about: instead of scoring a missing arm as 0, give
-- it a floor rank just past the fan-out --
--
--     COALESCE(f.vec_rank, %(fanout)s + 1)
--
-- This stops a strong single-arm hit being beaten by a mediocre both-arm hit.
-- It is not canonical RRF, so say so when you use it.
