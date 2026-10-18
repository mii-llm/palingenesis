"""Environment episodes outside training (evaluation, difficulty calibration, teacher traces) through the RL
pipeline's own rollout loop: the same prompt encoding, tool-call parsing, text protocols, harness profiles, stop
rules, truncation and environment rewards as an RL run, so an evaluation scores what training would see.

    from palingenesis.rl.episodes import run_episodes
    groups = run_episodes(engine, tok, rows, HarborEnv, {"profiles": {"opencode": 1, "pi": 1}}, k=4,
                          temperature=Sampling(0.7, 0.8, 20), chat_template_kwargs={"enable_thinking": False})
    # groups[i]: the k trajectories of rows[i] (t.reward, t.rewards, t.messages, t.finish, t.info)

max_completion_tokens: the generated tokens of a whole episode (all turns); default max_new_tokens x max_turns (the
context, max_model_len, still bounds it). engine: a rollout engine (palingenesis.opd.rollout: VLLMColocateRollout for vLLM in-process, or anything with
generate(prompts, max_new_tokens, temperature)); temperature: a float or an opd.rollout.Sampling (top_p / top_k;
streaming engines only). A group's k samples share the environment's per-group choices (a harness profile).
"""

from __future__ import annotations

from typing import Any, Callable


def run_episodes(
    engine,
    tokenizer,
    rows: list[dict],
    env: Callable[..., Any],
    env_args: dict | None = None,
    *,
    k: int = 1,
    temperature=1.0,
    chat_template_kwargs: dict | None = None,
    model: str | None = None,
    max_turns: int = 16,
    max_new_tokens: int = 1024,
    max_completion_tokens: int | None = None,
    max_model_len: int = 16384,
    max_concurrent: int = 0,
    tool_timeout: float = 600.0,
    max_tool_output_tokens: int = 1536,
    tool_parser: str = "auto",
) -> list[list[Any]]:
    import torch

    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.opd.trainer import student_stop_ids
    from palingenesis.rl.chat import ChatFormat
    from palingenesis.rl.config import RLConfig
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.env import EnvPool
    from palingenesis.rl.pipeline import RLPipeline

    if max_completion_tokens is None:  # the whole episode's generated tokens; the pipeline's 0 means one turn's
        max_completion_tokens = max_new_tokens * max_turns  # worth, which cuts multi-turn episodes short
    kwargs = dict(chat_template_kwargs or {})
    eot = end_of_turn_id(tokenizer, kwargs)
    stop_ids = student_stop_ids(tokenizer, model, [], kwargs) if model else (eot,)
    config = RLConfig()
    for key, value in {
        "model.policy": model or "episodes",
        "model.chat_template_kwargs": kwargs,
        "env.max_turns": max_turns,
        "env.tool_timeout": tool_timeout,
        "env.max_tool_output_tokens": max_tool_output_tokens,
        "env.tool_parser": tool_parser,
        "rollout.max_new_tokens": max_new_tokens,
        "rollout.max_completion_tokens": max_completion_tokens,
        "rollout.max_model_len": max_model_len,
    }.items():
        config.set(key, value)
    try:
        from palingenesis.rl.trainer import _think_tags

        chat = ChatFormat(tokenizer, eot, kwargs, _think_tags(tokenizer, kwargs))
    except Exception:  # noqa: BLE001 — a template without think tags
        chat = ChatFormat(tokenizer, eot, kwargs)
    pipeline = RLPipeline(
        tokenizer,
        chat,
        engine,
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler(rows),
        [],
        EnvPool(env, env_args or {}, max_concurrent),
        None,
        stop_ids,
    )
    try:
        return pipeline._await(pipeline._eval(rows, temperature, k))
    finally:
        pipeline.close()
