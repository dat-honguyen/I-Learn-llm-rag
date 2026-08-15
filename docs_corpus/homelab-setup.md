# Where this actually runs

This service runs as two containers on my home server, managed with Podman + Quadlet
(not docker-compose). One container runs Ollama and isn't reachable from outside the
box at all. Only this API container can talk to it, over an internal network. The API
container is the only thing Caddy proxies to, and Caddy is the only thing a Cloudflare
Tunnel exposes to the internet.

The public endpoint is rate-limited per IP and sits behind a shared password, mostly to
stop random bots from running up the CPU on a box that's also hosting a few other
things for me.
