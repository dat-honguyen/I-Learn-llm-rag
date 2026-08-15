from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    text = "one two three"
    assert chunk_text(text, chunk_size=500, overlap=50) == ["one two three"]


def test_long_text_splits_into_overlapping_chunks():
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) == 3
    assert chunks[0] == " ".join(words[0:50])
    assert chunks[1] == " ".join(words[40:90])
    assert chunks[2] == " ".join(words[80:120])
