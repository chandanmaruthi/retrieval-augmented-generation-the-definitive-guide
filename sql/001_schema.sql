-- ---------------------------------------------------------------------------
-- Core schema: documents, chunks, and one embedding table per vector space.
--
-- Applied by scripts/init_db.py. Safe to re-run.
--
-- Targets the Postgres in docker-compose.yml. Any Postgres 15+ with pgvector
-- available will do -- NULLS NOT DISTINCT on the edges table is the only
-- thing here that needs 15 rather than something older.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;


-- ---------------------------------------------------------------------------
-- documents
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS documents (
    id          bigserial PRIMARY KEY,
    slug        text NOT NULL UNIQUE,
    title       text NOT NULL,
    author      text,
    source      text,
    sha256      char(64),
    n_words     integer,
    created_at  timestamptz NOT NULL DEFAULT now()
);


-- ---------------------------------------------------------------------------
-- chunks
--
-- Every chunking strategy writes into this one table, discriminated by the
-- `strategy` column. Comparing thirteen strategies is then a GROUP BY rather
-- than thirteen schemas, and a query can be pointed at a different strategy
-- by changing one bound parameter.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    strategy     text NOT NULL,
    ordinal      integer NOT NULL,

    -- What the generator reads.
    body         text NOT NULL,
    -- What was embedded, when that differs: a header-prefixed passage, a bare
    -- sentence, a generated question. NULL means "same as body".
    embed_body   text,
    -- The larger passage to return on a hit, for parent-child and
    -- sentence-window strategies. NULL means "same as body".
    parent_body  text,

    story        text,
    section      text,

    -- Access control, carried from extraction. The book is explicit that this
    -- belongs in the index rather than in a post-retrieval filter, and
    -- sql/002_rls.sql is where it starts being enforced.
    acl          text[] NOT NULL DEFAULT ARRAY['public'],

    -- Streaming ingestion key: re-ingesting unchanged text is a no-op.
    content_hash char(64) NOT NULL,

    meta         jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now(),

    -- The lexical index lives here, on `chunks`, and NOT on the embedding
    -- tables. It is independent of which embedding model you use, so swapping
    -- models never rebuilds it.
    --
    -- to_tsvector(regconfig_literal, text) is IMMUTABLE, which is what makes a
    -- generated column legal. Writing to_tsvector('english', ...) without the
    -- explicit regconfig is only STABLE and Postgres will reject it here.
    tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english'::regconfig, coalesce(body, ''))
    ) STORED,

    UNIQUE (document_id, strategy, ordinal),
    -- Referenced by the composite foreign key on the embedding tables.
    UNIQUE (id, strategy)
);


-- ---------------------------------------------------------------------------
-- embeddings, one table per vector space
--
-- pgvector columns are fixed width and an ANN index is built over exactly one
-- vector space, so a table per space matches the physical reality. MiniLM is
-- 384 dimensions and text-embedding-3-small is 1536; they cannot share a
-- column and should not share an index.
--
-- The tempting alternative -- a single unconstrained `vector` column -- is a
-- trap. pgvector accepts it and then refuses to index it, so you get
-- sequential scans forever with no error to tell you why.
--
-- `strategy` is denormalised onto these tables on purpose. See 003_indexes.sql.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS embeddings_384 (
    chunk_id   bigint NOT NULL,
    model      text   NOT NULL,
    strategy   text   NOT NULL,
    vec        vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model),
    -- The composite key keeps the denormalised `strategy` honest without a
    -- trigger: it cannot disagree with the chunk it points at.
    FOREIGN KEY (chunk_id, strategy) REFERENCES chunks (id, strategy) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS embeddings_1536 (
    chunk_id   bigint NOT NULL,
    model      text   NOT NULL,
    strategy   text   NOT NULL,
    vec        vector(1536) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, model),
    FOREIGN KEY (chunk_id, strategy) REFERENCES chunks (id, strategy) ON DELETE CASCADE
);


-- ---------------------------------------------------------------------------
-- graph: entities and relations, for examples 25 and 26
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS entities (
    id      bigserial PRIMARY KEY,
    name    text NOT NULL,
    kind    text NOT NULL,
    aliases text[] NOT NULL DEFAULT '{}',
    meta    jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (name, kind)
);

CREATE TABLE IF NOT EXISTS edges (
    id                bigserial PRIMARY KEY,
    src_id            bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    dst_id            bigint NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation          text NOT NULL,
    weight            real NOT NULL DEFAULT 1.0,
    evidence_chunk_id bigint REFERENCES chunks(id) ON DELETE SET NULL,
    -- Without NULLS NOT DISTINCT, two edges that differ only by a NULL
    -- evidence id are considered distinct and this constraint silently
    -- permits duplicates. Postgres 15+.
    UNIQUE NULLS NOT DISTINCT (src_id, dst_id, relation, evidence_chunk_id)
);


-- ---------------------------------------------------------------------------
-- memory, feedback and evaluation: examples 30, 31, 36, 37
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS memory (
    id           bigserial PRIMARY KEY,
    session_id   text NOT NULL,
    kind         text NOT NULL CHECK (kind IN ('turn', 'fact', 'summary')),
    role         text CHECK (role IN ('user', 'assistant', 'system')),
    content      text NOT NULL,
    importance   real NOT NULL DEFAULT 0.5,
    -- Deliberately the opposite pattern from `chunks`: this table holds
    -- hundreds of rows, not hundreds of thousands, so there is no ANN index
    -- and nullable columns for both vector spaces cost nothing.
    vec_384      vector(384),
    vec_1536     vector(1536),
    model        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    last_used_at timestamptz,
    use_count    integer NOT NULL DEFAULT 0,
    expires_at   timestamptz,
    CONSTRAINT memory_one_vector CHECK (num_nonnulls(vec_384, vec_1536) <= 1)
);

CREATE TABLE IF NOT EXISTS feedback (
    id         bigserial PRIMARY KEY,
    query      text NOT NULL,
    chunk_id   bigint REFERENCES chunks(id) ON DELETE SET NULL,
    rank       integer,
    rating     smallint CHECK (rating BETWEEN 1 AND 5),
    verdict    smallint CHECK (verdict IN (-1, 0, 1)),
    correction text,
    reviewer   text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS query_log (
    id              bigserial PRIMARY KEY,
    query           text NOT NULL,
    strategy        text,
    retriever       text,
    k               integer,
    latency_ms      integer,
    retrieved_ids   bigint[],
    grounded        boolean,
    hallucinated    boolean,
    prompt_tokens   integer,
    answer_tokens   integer,
    created_at      timestamptz NOT NULL DEFAULT now()
);
