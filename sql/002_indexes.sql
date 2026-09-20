-- ---------------------------------------------------------------------------
-- Indexes.
--
-- Read the comments before copying any of this. Several of these indexes are
-- here to be *measured* in example 20, and the honest conclusion on a corpus
-- this size is that the ANN indexes do not earn their keep.
-- ---------------------------------------------------------------------------


-- ======================= lexical / metadata side ===========================

-- Serves the full-text arm of the hybrid query in example 19.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

-- Serves the ACL overlap test in the row-level security policy. Array
-- containment (&&) needs a GIN index; a btree does nothing for it.
CREATE INDEX IF NOT EXISTS chunks_acl_gin ON chunks USING gin (acl);

CREATE INDEX IF NOT EXISTS chunks_strategy ON chunks (strategy);
CREATE INDEX IF NOT EXISTS chunks_story ON chunks (story);
CREATE INDEX IF NOT EXISTS chunks_doc_strategy_ord ON chunks (document_id, strategy, ordinal);

-- Streaming ingestion: look up whether this exact text is already indexed, so
-- an unchanged chunk can skip re-embedding (example 29).
--
-- Deliberately NOT unique. The obvious version of this index is
--
--     CREATE UNIQUE INDEX ... ON chunks (document_id, strategy, content_hash)
--
-- and it is wrong, because identical chunk text within one document is
-- legitimate. Sentence-window chunking (example 07) indexes every sentence
-- separately, and this corpus contains '"Indeed!"' more than once. The unique
-- version fails at ingest time with a constraint violation that looks like a
-- bug in the chunker rather than a bug in the schema.
--
-- Identity is (document_id, strategy, ordinal), which is already UNIQUE on
-- the table. The hash answers "has this text changed?", not "is this row
-- distinct?".
CREATE INDEX IF NOT EXISTS chunks_hash_idx
    ON chunks (document_id, strategy, content_hash);


-- ============================ dense / ANN side =============================
--
-- IMPORTANT: at this corpus's scale there is no ANN index below, on purpose.
--
-- Each strategy produces roughly 400-7,000 chunks. An exact scan over a
-- 7,000 x 384 matrix is a single pass that Postgres does in a couple of
-- milliseconds, with perfect recall. An HNSW index over the same data is
-- slower to query (it has to be built first, and it still reads heap tuples)
-- and returns approximate results. Example 20 builds one, measures both, and
-- reports the difference rather than assuming it.
--
-- The statements below are what you WOULD run at a scale where it matters.
-- scripts/init_db.py applies them only with --ann, so the default setup
-- stays honest.


-- --- HNSW, global ----------------------------------------------------------
--
-- m = 16, ef_construction = 64 are pgvector's defaults and a reasonable start.
-- Build time is dominated by maintenance_work_mem; raise it for the session
-- rather than globally:  SET maintenance_work_mem = '512MB';
-- On a memory-capped instance that ceiling is what decides how long an index
-- build takes, so check it before waiting twenty minutes for one.
--
-- CREATE INDEX emb384_hnsw ON embeddings_384
--     USING hnsw (vec vector_cosine_ops) WITH (m = 16, ef_construction = 64);


-- --- HNSW, partial by strategy ---------------------------------------------
--
-- This is the subtle one, and the reason `strategy` is denormalised onto
-- embeddings_384 at all.
--
-- A global HNSW index combined with `WHERE strategy = 'heading'` is
-- POST-filtered: pgvector walks the graph, returns ef_search candidates from
-- across every strategy, and only then applies the filter. Ask for 5 and you
-- may get 2, with no error. It is the most common pgvector surprise there is.
--
-- A partial index puts the predicate inside the index, so every candidate it
-- returns already satisfies the filter:
--
-- CREATE INDEX emb384_hnsw_heading ON embeddings_384
--     USING hnsw (vec vector_cosine_ops)
--     WHERE strategy = 'heading';
--
-- pgvector 0.8+ offers an alternative if you would rather keep one index:
--     SET LOCAL hnsw.iterative_scan = 'relaxed_order';
--     SET LOCAL hnsw.ef_search = 200;


-- --- IVFFlat ---------------------------------------------------------------
--
-- Cheaper to build, worse recall, and it has a trap of its own: IVFFlat
-- clusters the data at build time, so building it on an empty table produces
-- meaningless centroids and permanently bad recall. It must be built AFTER
-- the data is loaded.
--
-- lists ~= rows/1000 for under a million rows, with a floor around 10.
-- probes ~= sqrt(lists) at query time.
--
-- CREATE INDEX emb384_ivf ON embeddings_384
--     USING ivfflat (vec vector_cosine_ops) WITH (lists = 32);
-- SET LOCAL ivfflat.probes = 6;


-- --- Inner product instead of cosine ---------------------------------------
--
-- ragkit's embedders all return L2-normalised vectors. For unit vectors,
-- cosine similarity and inner product give identical rankings, and inner
-- product is slightly cheaper because it skips the norm.
--
-- CREATE INDEX emb384_hnsw_ip ON embeddings_384
--     USING hnsw (vec vector_ip_ops);


-- --- halfvec for the 1536-dimension space ----------------------------------
--
-- Halves index size at a very small recall cost. Worth it above 1000
-- dimensions, which is why it is suggested here and not for the 384 table.
--
-- CREATE INDEX emb1536_hnsw_half ON embeddings_1536
--     USING hnsw ((vec::halfvec(1536)) halfvec_cosine_ops);


-- ============================== graph side =================================

CREATE INDEX IF NOT EXISTS edges_src ON edges (src_id);
CREATE INDEX IF NOT EXISTS edges_dst ON edges (dst_id);
CREATE INDEX IF NOT EXISTS entities_name ON entities (lower(name));


-- ========================= memory / logging side ===========================

CREATE INDEX IF NOT EXISTS memory_session ON memory (session_id, created_at DESC);

-- NOT this -- now() is not IMMUTABLE and CREATE INDEX rejects it:
--   CREATE INDEX memory_live ON memory (session_id) WHERE expires_at > now();
-- Index the column instead and filter in the query.
CREATE INDEX IF NOT EXISTS memory_expires ON memory (expires_at)
    WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS query_log_created ON query_log (created_at DESC);
