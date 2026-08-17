from app import store

EMBED_DIM = 8


def make_embedding(seed: float) -> list[float]:
    return [seed] * EMBED_DIM


def test_insert_and_top_k_returns_closest_chunk(tmp_path):
    conn = store.init_db(str(tmp_path / "vectors.db"), dim=EMBED_DIM)

    store.insert_chunk(conn, "doc1", 0, "chunk about cats", "hash1", make_embedding(1.0))
    store.insert_chunk(conn, "doc1", 1, "chunk about dogs", "hash1", make_embedding(5.0))
    store.commit(conn)

    results = store.top_k(conn, make_embedding(1.1), k=1)

    assert len(results) == 1
    assert results[0][0] == "chunk about cats"
    assert results[0][1] == "doc1"


def test_get_doc_hash_returns_none_when_absent(tmp_path):
    conn = store.init_db(str(tmp_path / "vectors.db"), dim=EMBED_DIM)
    assert store.get_doc_hash(conn, "missing-doc") is None


def test_delete_doc_removes_its_chunks(tmp_path):
    conn = store.init_db(str(tmp_path / "vectors.db"), dim=EMBED_DIM)
    store.insert_chunk(conn, "doc1", 0, "text a", "hash1", make_embedding(1.0))
    store.commit(conn)

    store.delete_doc(conn, "doc1")
    store.commit(conn)

    assert store.get_doc_hash(conn, "doc1") is None
