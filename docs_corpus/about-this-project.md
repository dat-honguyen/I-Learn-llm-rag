# Why I built this

I wanted to actually learn how RAG works instead of just reading about it, so I built
a small end-to-end version myself: a tiny FastAPI service, a local model running on my
own homelab box, and a vector store holding chunks of these exact notes you're reading.

It runs on a Ryzen 5 8600G with 16GB of RAM. No GPU, no cloud API keys, no credit card.
Everything you ask it gets answered by actually searching these markdown files first,
then handing the relevant pieces to the model as context. If the answer isn't in these
notes, it's supposed to say so instead of making something up.
