-- ===========================================================================
-- Hybrid retrieval, linear weighting.
--
-- The centrepiece query of this repository. Its *shape* matters more than its
-- arithmetic, and three rules drive that shape:
--
--  1. Each arm is its own candidate-generation query, with its own ORDER BY
--     and its own LIMIT, so each can use its own index. Fuse first and order
--     by a combined score and NEITHER index can be used -- the planner falls
--     back to reading every row.
--
--  2. The vector arm must ORDER BY a bare `vec <=> query`, ascending. Writing
--     `ORDER BY (1 - (vec <=> query)) DESC` computes exactly the same ranking
--     and silently disables the HNSW index, because the operator is no longer
--     the bare expression the index is defined on.
--
--  3. Ranks are assigned in an outer query, after the LIMIT has applied. A
--     window function in the same SELECT as the LIMIT is evaluated over every
--     matching row first, which forces a full sort.
--
-- Parameters (psycopg named):
--   qvec, qtext, strategy, model, fanout, k, w_vec, w_txt
-- ===========================================================================

WITH vec_candidates AS (
    SELECT e.chunk_id,
           e.vec <=> %(qvec)s::vector AS distance
    FROM   embeddings_384 e
    WHERE  e.strategy = %(strategy)s
      AND  e.model    = %(model)s
    ORDER  BY e.vec <=> %(qvec)s::vector          -- rule 2: bare operator, ASC
    LIMIT  %(fanout)s                             -- rule 1: own limit
),
vec AS (                                          -- rule 3: rank after LIMIT
    SELECT chunk_id,
           distance,
           1.0 - distance                        AS vec_score,
           ROW_NUMBER() OVER (ORDER BY distance) AS vec_rank
    FROM   vec_candidates
),

txt_candidates AS (
    SELECT c.id AS chunk_id,
           -- Normalisation flag 32 divides by (rank + 1), bounding the score
           -- to (0, 1). Without it ts_rank returns an unbounded number whose
           -- magnitude depends on document length, which makes the weighting
           -- below meaningless.
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
    -- FULL OUTER JOIN, not INNER. A chunk found by only one arm must survive:
    -- a question phrased in words absent from the corpus produces an empty
    -- lexical arm, and an INNER JOIN would then return nothing at all rather
    -- than falling back on the vector results.
    --
    -- USING (chunk_id) merges the join column so it is never NULL on either
    -- side, which is why the SELECT below can reference it unqualified.
    SELECT chunk_id,
           COALESCE(v.vec_score, 0.0) AS vec_score,
           COALESCE(t.ts_score,  0.0) AS txt_score,
           v.vec_rank,
           t.txt_rank
    FROM   vec v FULL OUTER JOIN txt t USING (chunk_id)
),

-- ---------------------------------------------------------------------------
-- The step everybody skips, and the reason naive linear fusion is really just
-- vector search wearing a hat.
--
-- Cosine similarity on this corpus lands around 0.2-0.6. ts_rank_cd with
-- normalisation 32 lands around 0.01-0.10. Combine them at w_vec = w_txt = 0.5
-- and the lexical arm contributes a few percent of the total: you have built
-- a hybrid retriever that ignores half its input.
--
-- Min-max over the candidate set puts both arms on [0, 1] so the weights mean
-- what they say. NULLIF guards the case where every candidate scored
-- identically, which would otherwise divide by zero.
-- ---------------------------------------------------------------------------
bounds AS (
    SELECT MIN(vec_score) AS v_lo,
           NULLIF(MAX(vec_score) - MIN(vec_score), 0) AS v_span,
           MIN(txt_score) AS t_lo,
           NULLIF(MAX(txt_score) - MIN(txt_score), 0) AS t_span
    FROM   fused
),

scored AS (
    SELECT f.chunk_id,
           f.vec_score, f.txt_score, f.vec_rank, f.txt_rank,
           COALESCE((f.vec_score - b.v_lo) / b.v_span, 0.0) AS vec_norm,
           COALESCE((f.txt_score - b.t_lo) / b.t_span, 0.0) AS txt_norm
    FROM   fused f CROSS JOIN bounds b
)

SELECT c.id AS chunk_id,
       c.story,
       c.section,
       c.body,
       c.parent_body,
       s.vec_score,
       s.txt_score,
       s.vec_norm,
       s.txt_norm,
       s.vec_rank,
       s.txt_rank,
       (%(w_vec)s * s.vec_norm + %(w_txt)s * s.txt_norm) AS score
FROM   scored s
JOIN   chunks c ON c.id = s.chunk_id
ORDER  BY score DESC, s.chunk_id
LIMIT  %(k)s;
