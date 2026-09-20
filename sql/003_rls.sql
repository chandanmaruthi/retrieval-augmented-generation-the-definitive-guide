-- ---------------------------------------------------------------------------
-- Row-level security, for example 35.
--
-- The book is unambiguous about this:
--
--   "a system that retrieves first and redacts afterwards has already leaked
--    the document into the model's context"
--
-- So access control is enforced by the database on the way out, not by Python
-- on the way back. A filtered-out row is a row the retrieval never saw, never
-- ranked, and never put in a prompt.
-- ---------------------------------------------------------------------------


-- The identity the examples read as. NOLOGIN: it is assumed with SET ROLE
-- from the connection ragkit already has, rather than connected to directly.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_reader') THEN
        CREATE ROLE rag_reader NOLOGIN;
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO rag_reader;
GRANT SELECT ON chunks, embeddings_384, embeddings_1536, documents TO rag_reader;


ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;

-- Without FORCE, the table's OWNER bypasses every policy below.
--
-- This matters more than it looks. DATABASE_URL almost always connects as the
-- owner -- it certainly does with docker-compose.yml. Omit
-- FORCE and example 35 runs green while proving nothing at all: every policy
-- is in place, and every one of them is being ignored.
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;


-- Policies are OR-ed together. Anything tagged 'public' is readable by anyone.
DROP POLICY IF EXISTS chunks_read_public ON chunks;
CREATE POLICY chunks_read_public ON chunks
    FOR SELECT
    USING ('public' = ANY (acl));


-- Everything else depends on the caller's groups, which arrive as a session
-- setting. `&&` is array overlap, served by chunks_acl_gin.
--
-- current_setting(..., true) returns NULL rather than raising when the
-- setting was never set, which is what makes this safe for an anonymous
-- connection: NULL groups match nothing, so only the public policy applies.
DROP POLICY IF EXISTS chunks_read_group ON chunks;
CREATE POLICY chunks_read_group ON chunks
    FOR SELECT
    USING (
        acl && string_to_array(coalesce(current_setting('app.groups', true), ''), ',')
    );


-- Writes are unrestricted for the owner, which is how ingestion works.
DROP POLICY IF EXISTS chunks_write ON chunks;
CREATE POLICY chunks_write ON chunks
    FOR ALL
    TO CURRENT_USER
    USING (true)
    WITH CHECK (true);


-- ---------------------------------------------------------------------------
-- The vector table leaks too.
--
-- An unprotected embeddings table will happily tell you that a chunk exists
-- and how similar it is to your query, even when the policy above stops you
-- reading its text. Similarity scores over a corpus you cannot read are a
-- real disclosure: run enough queries and you can reconstruct a great deal.
-- ---------------------------------------------------------------------------

ALTER TABLE embeddings_384 ENABLE ROW LEVEL SECURITY;
ALTER TABLE embeddings_384 FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS emb384_follows_chunks ON embeddings_384;
CREATE POLICY emb384_follows_chunks ON embeddings_384
    FOR SELECT
    USING (EXISTS (SELECT 1 FROM chunks c WHERE c.id = embeddings_384.chunk_id));

-- That policy is correct and deliberately slow. The sub-select runs per
-- candidate row AFTER the ANN index has returned, which is post-filtering
-- again -- the same trap as the partial-index note in 002_indexes.sql.
--
-- The production fix is to denormalise `acl` onto embeddings_384 and write
-- the policy as a direct array overlap, so a GIN index can serve it:
--
--   ALTER TABLE embeddings_384 ADD COLUMN acl text[] NOT NULL DEFAULT '{public}';
--   CREATE INDEX emb384_acl_gin ON embeddings_384 USING gin (acl);
--   CREATE POLICY emb384_acl ON embeddings_384 FOR SELECT
--       USING (acl && string_to_array(coalesce(current_setting('app.groups', true), ''), ','));
--
-- Example 35 measures both and shows the difference.


-- ---------------------------------------------------------------------------
-- Setting the caller's identity from Python
--
--   WRONG -- SET does not take bind parameters, this is a syntax error:
--       cur.execute("SET app.groups = %s", [groups])
--
--   RIGHT:
--       cur.execute("SELECT set_config('app.groups', %s, false)", [",".join(groups)])
--
-- ragkit/pg.py does the second. The distinction costs people an afternoon.
-- ---------------------------------------------------------------------------
