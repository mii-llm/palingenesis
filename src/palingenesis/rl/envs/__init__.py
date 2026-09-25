"""Adapters that make other environment standards palingenesis environments.

  openenv.OpenEnvAdapter     OpenEnv (Meta PyTorch / Hugging Face) servers: remote over
                             WebSocket, or the server class in-process; MCP-tool or step envs
  nemo_gym.NemoGymAdapter    NVIDIA NeMo Gym resources servers (the Nemotron-RL datasets'
                             verifiers): seed_session, one route per tool, verify

Each implements the ordinary environment protocol (palingenesis.rl.env), so the trainer
needs no special case:

    env:
      type: palingenesis.rl.envs.openenv:OpenEnvAdapter
      max_concurrent: 64              # at most the server's concurrent sessions
      args: {base_url: http://localhost:8001, reset_fields: [seed]}
"""
