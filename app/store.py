import sqlite3
import struct

import sqlite_vec

DEFAULT_DIM = 768


def _serialize(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def init_db(path: str, dim: int = DEFAULT_DIM) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            UNIQUE(doc_id, chunk_index)
        )
        """
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])"
    )
    conn.commit()
    return conn


def get_doc_hash(conn: sqlite3.Connection, doc_id: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM chunks WHERE doc_id = ? LIMIT 1", (doc_id,)
    ).fetchone()
    return row[0] if row else None


def delete_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
    ]
    for chunk_id in ids:
        conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def insert_chunk(
    conn: sqlite3.Connection,
    doc_id: str,
    chunk_index: int,
    text: str,
    content_hash: str,
    embedding: list[float],
) -> None:
    cursor = conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, content_hash) VALUES (?, ?, ?, ?)",
        (doc_id, chunk_index, text, content_hash),
    )
    chunk_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        (chunk_id, _serialize(embedding)),
    )


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()


def top_k(conn: sqlite3.Connection, embedding: list[float], k: int) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT c.text, v.distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (_serialize(embedding), k),
    ).fetchall()
    return [(text, distance) for text, distance in rows]
