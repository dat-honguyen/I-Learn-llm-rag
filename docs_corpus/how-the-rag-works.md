# How the retrieval actually works

When you ask a question, three things happen in order:

1. The question gets turned into a vector (an embedding) using Ollama's
   `nomic-embed-text` model.
2. That vector gets compared against vectors for every chunk of these notes, stored in
   a local SQLite file using the `sqlite-vec` extension. The closest few chunks win.
3. Those chunks get pasted into a prompt along with your question, and the whole thing
   gets sent to a small local chat model (`llama3.1:8b-instruct`, quantized down to fit
   comfortably in RAM alongside everything else running on the box).

I picked `sqlite-vec` over a "real" vector database on purpose. There are maybe 20-30
chunks total across these notes. Running a separate Postgres+pgvector container for
that would be solving a problem I don't have.
