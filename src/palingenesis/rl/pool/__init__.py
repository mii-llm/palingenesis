"""Rollout engines in their own processes, on their own GPUs or nodes (rollout.backend: remote).

    python -m palingenesis.rl.pool.worker --model Qwen/Qwen3.5-0.8B --port 7000      # on each engine GPU
    rollout: {backend: remote, engines: ["node2:7000", "node2:7001"], max_staleness: 1}

protocol   framed messages with raw tensor payloads over a TCP stream
worker     an engine process: generation requests in its own continuous batch, weights staged while it keeps
           generating and applied between two engine steps
client     the trainer's side: requests spread over the engines, weight pushes that never wait for the engines

What crosses the wire for a weight update is chosen by the trainer (rollout.push): every weight ("full"), or with
LoRA adapters only their factors, megabytes, either folded into the base by each engine ("fold": any engine, one
policy per engine) or loaded as named adapters that share an engine's batch ("native": vLLM's LoRA serving,
several roles' policies in one engine).
"""
