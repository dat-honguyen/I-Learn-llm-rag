# Deploying to the homelab box

1. Copy `deploy/llm-rag.network`, `deploy/ollama.container`, and
   `deploy/llm-rag-api.container` into `~/.config/containers/systemd/` on `homelab`.
2. Replace the `APP_PASSWORD` placeholder in `llm-rag-api.container` with a real value
   (don't commit it).
3. `systemctl --user daemon-reload`
4. `systemctl --user start ollama`
5. `podman exec ollama ollama pull llama3.2:3b-instruct-q4_K_M`
6. `podman exec ollama ollama pull nomic-embed-text`
7. Build the API image (or let the self-hosted Actions runner do it via
   `.github/workflows/ci.yml`), then `systemctl --user start llm-rag-api`.
8. Append `deploy/caddy-llm-rag.snippet`'s contents to `~/caddy/Caddyfile`, then
   `systemctl --user restart caddy`.
9. `cloudflared tunnel route dns mydebian-sv llm.datisa.dev` (check
   `/etc/cloudflared/config.yml` first, the ingress rule is likely already generic).
10. Smoke test: `curl -X POST https://llm.datisa.dev/chat -d '{"question": "...", "password": "..."}'`.
