"""
Benchmarks a chat model against this project's live Ollama instance.

Run inside the llm-rag-api container, where OLLAMA_URL/httpx are already available:

    podman exec llm-rag-api python3 /tmp/model_bench.py --model llama3.2:3b-instruct-q4_K_M

Copy this file in first: podman cp benchmarks/model_bench.py llm-rag-api:/tmp/model_bench.py

Uses /api/chat with stream=False so Ollama reports exact eval_count/eval_duration
per request — the same numbers `ollama ps` and the Ollama logs use internally, not an
estimate from wall-clock timing.
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx

QUESTIONS = [
    "who is dat",
    "who are you",
    "what is this",
    "what do you do",
    "why did Dat pick sqlite-vec instead of a real vector database?",
    "what database does the RAG project use?",
    "where did the author work at Topicus Healthcare?",
    "what tools does Dat use day to day?",
    "how does the retrieval step work?",
    "what hardware does this run on?",
]


async def run_one(client: httpx.AsyncClient, ollama_url: str, model: str, question: str) -> dict:
    start = time.perf_counter()
    response = await client.post(
        f"{ollama_url}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "stream": False,
        },
        timeout=180.0,
    )
    wall_seconds = time.perf_counter() - start
    payload = response.json()

    eval_count = payload.get("eval_count", 0)
    eval_duration_s = payload.get("eval_duration", 0) / 1e9
    tokens_per_sec = eval_count / eval_duration_s if eval_duration_s else 0.0

    return {
        "question": question,
        "wall_seconds": round(wall_seconds, 2),
        "eval_count": eval_count,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "prompt_eval_count": payload.get("prompt_eval_count", 0),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ollama-url", default="http://ollama:11434")
    parser.add_argument("--out", default="/tmp/bench_result.json")
    args = parser.parse_args()

    results = []
    async with httpx.AsyncClient() as client:
        for question in QUESTIONS:
            result = await run_one(client, args.ollama_url, args.model, question)
            print(f"{result['wall_seconds']:>6.2f}s  {result['tokens_per_sec']:>6.2f} tok/s  {question}")
            results.append(result)

    wall_times = [r["wall_seconds"] for r in results]
    speeds = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"]]

    summary = {
        "model": args.model,
        "questions_run": len(results),
        "wall_seconds_mean": round(statistics.mean(wall_times), 2),
        "wall_seconds_median": round(statistics.median(wall_times), 2),
        "wall_seconds_max": round(max(wall_times), 2),
        "tokens_per_sec_mean": round(statistics.mean(speeds), 2) if speeds else 0,
        "results": results,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nmean {summary['wall_seconds_mean']}s, median {summary['wall_seconds_median']}s, "
          f"max {summary['wall_seconds_max']}s, mean {summary['tokens_per_sec_mean']} tok/s")
    print(f"written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
