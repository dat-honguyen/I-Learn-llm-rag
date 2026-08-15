import hashlib
from pathlib import Path
from typing import Awaitable, Callable

from . import store
from .chunking import chunk_text


async def ingest_docs(
    docs_dir: Path,
    conn,
    embed_fn: Callable[[str], Awaitable[list[float]]],
) -> int:
    ingested = 0
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = path.stem

        if store.get_doc_hash(conn, doc_id) == content_hash:
            continue

        store.delete_doc(conn, doc_id)
        for index, chunk in enumerate(chunk_text(text)):
            embedding = await embed_fn(chunk)
            store.insert_chunk(conn, doc_id, index, chunk, content_hash, embedding)
        store.commit(conn)
        ingested += 1

    return ingested
