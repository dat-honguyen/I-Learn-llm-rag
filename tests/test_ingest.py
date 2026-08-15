from pathlib import Path

from app import store
from app.ingest import ingest_docs


async def fake_embed(text: str) -> list[float]:
    return [float(len(text))] * 8


async def test_ingest_embeds_new_docs(tmp_path):
    docs_dir = tmp_path / "docs_corpus"
    docs_dir.mkdir()
    (docs_dir / "note.md").write_text("hello world this is a note", encoding="utf-8")

    conn = store.init_db(str(tmp_path / "vectors.db"), dim=8)

    count = await ingest_docs(docs_dir, conn, fake_embed)

    assert count == 1
    assert store.get_doc_hash(conn, "note") is not None


async def test_ingest_skips_unchanged_docs_on_second_run(tmp_path):
    docs_dir = tmp_path / "docs_corpus"
    docs_dir.mkdir()
    (docs_dir / "note.md").write_text("same content", encoding="utf-8")

    conn = store.init_db(str(tmp_path / "vectors.db"), dim=8)

    first = await ingest_docs(docs_dir, conn, fake_embed)
    second = await ingest_docs(docs_dir, conn, fake_embed)

    assert first == 1
    assert second == 0


async def test_ingest_reembeds_changed_docs(tmp_path):
    docs_dir = tmp_path / "docs_corpus"
    docs_dir.mkdir()
    doc_path = docs_dir / "note.md"
    doc_path.write_text("version one", encoding="utf-8")

    conn = store.init_db(str(tmp_path / "vectors.db"), dim=8)
    await ingest_docs(docs_dir, conn, fake_embed)

    doc_path.write_text("version two, changed", encoding="utf-8")
    second = await ingest_docs(docs_dir, conn, fake_embed)

    assert second == 1
