"""Images in multi-turn RL: tool results with images (inside the tool message, or in a user message
after it), rollout and training on identical tokens, positions and pixels, and dropping old
screenshots without retraining any turn on a context it did not see. Tiny Qwen3.5 models, CPU."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from multimodal_helpers import save_tiny, tiny_image, tiny_image_processor, tiny_qwen3_5, tiny_tokenizer

from palingenesis import multimodal
from palingenesis.multimodal import VisionProcessor
from palingenesis.opd.rollout import HFRollout, Rollout
from palingenesis.rl.chat import ChatFormat
from palingenesis.rl.config import RLConfig, RLConfigError
from palingenesis.rl.env import EnvPool, ToolOutput

KWARGS = {"enable_thinking": False}
CALL = "Looking.\n<tool_call>\n<function=screenshot>\n</function>\n</tool_call>"


def _tok(vision: bool = True):
    tok = tiny_tokenizer()
    if vision:
        multimodal.attach(tok, VisionProcessor(tiny_image_processor(), tok, "qwen3_5"))
    return tok


class ScreenEnv:
    """A computer-use style environment: every call returns a screenshot of a new size."""

    sizes = ((16, 24), (24, 16), (16, 16), (8, 24))

    def __init__(self, text_only: bool = False):
        self.text_only = text_only

    def reset(self, **row):
        self.step = 0

    def screenshot(self) -> ToolOutput:
        """Take a screenshot of the screen."""
        self.step += 1
        size = self.sizes[(self.step - 1) % len(self.sizes)]
        return ToolOutput("" if self.text_only else f"screen {self.step}", [tiny_image(*size, self.step)])


class ActingEnv(ScreenEnv):
    """A text-protocol variant: whatever the policy writes is an action, so a random policy plays on."""

    def text_actions(self, content):
        return [("screenshot", {})]


class ScriptedEngine:
    """Answers from a script (turn index -> text) and records the prompts and images it was given."""

    def __init__(self, tok, eot, script):
        self.tok, self.eot, self.script, self.version, self.seen = tok, eot, script, 0, []

    def generate(self, prompts, max_new_tokens, temperature, images=None):
        out = []
        for i, prompt in enumerate(prompts):
            self.seen.append((list(prompt), list(images[i] or []) if images else []))
            turn = self.tok.decode(prompt).count("<|im_start|>assistant") - 1
            ids = self.tok.encode(self.script[min(turn, len(self.script) - 1)], add_special_tokens=False) + [self.eot]
            out.append(Rollout(ids, [-0.25] * len(ids), "stop", self.version))
        return out

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def _pipeline(tok, engine, env, **options):
    from palingenesis.opd.orchestrator import PublishedWeights
    from palingenesis.opd.teachers import end_of_turn_id
    from palingenesis.rl.data import PromptSampler
    from palingenesis.rl.pipeline import RLPipeline

    eot = end_of_turn_id(tok, KWARGS)
    config = RLConfig()
    for key, value in {
        "model.policy": "x",
        "model.chat_template_kwargs": KWARGS,
        "model.multimodal": multimodal.vision_of(tok) is not None,
        "env.max_turns": 4,
        "rollout.group_size": 1,
        "rollout.max_new_tokens": 64,
        "rollout.max_completion_tokens": 512,
        "rollout.max_model_len": 2048,
        **options,
    }.items():
        config.set(key, value)
    return RLPipeline(
        tok,
        ChatFormat(tok, eot, KWARGS),
        engine(eot) if callable(engine) else engine,
        PublishedWeights(torch.nn.Linear(1, 1)),
        config,
        PromptSampler([PROMPT]),
        [],
        EnvPool(env),
        None,
        (eot,),
    )


def _run(pipeline, rows, temperature=1.0, size=1):
    try:
        return pipeline._await(pipeline._eval(rows, temperature, size))
    finally:
        pipeline.close()


PROMPT = {"prompt": "Open the settings."}


# ----------------------------------------------------------------- where tool images go


def test_image_tool_result_inside_the_tool_message():
    tok = _tok()
    vision = multimodal.vision_of(tok)
    engine = ScriptedEngine(tok, tok.eos_token_id, [CALL, CALL, "Done."])
    (group,) = _run(_pipeline(tok, engine, ScreenEnv), [PROMPT])
    t = group[0]
    assert t.finish == "stop" and t.turns == 3 and t.tool_calls == 2, t.info.get("error")
    assert [m["role"] for m in t.messages] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    tool = t.messages[2]["content"]
    assert tool[0] == {"type": "text", "text": "screen 1"} and tool[1]["type"] == "image"
    # one placeholder per image in the tokens, inside the tool response the template renders
    context = tok.decode([x for x, m in zip(t.tokens, t.mask) if not m])
    assert context.count("<tool_response>\nscreen 1<|vision_start|><|image_pad|><|vision_end|>\n</tool_response>") == 1
    assert vision.count_images(t.prompt_ids + t.tokens) == len(t.images) == 2
    assert t.image_tokens == [vision.num_tokens(16, 24), vision.num_tokens(24, 16)]
    # image placeholders are context, never trained
    assert not any(m for x, m in zip(t.tokens, t.mask) if x == vision.image_token_id)
    # the engine saw the images of the context so far at each turn, one per placeholder
    assert [len(images) for _, images in engine.seen] == [0, 1, 2]
    assert all(vision.count_images(prompt) == len(images) for prompt, images in engine.seen)
    # the policy reads the expanded sequence: that is what counts against the context limit
    assert t.context_tokens == len(t.prompt_ids) + len(t.tokens) + sum(n - 1 for n in t.image_tokens)


def test_image_tool_result_in_a_following_user_message():
    """What OpenAI-style chat completions allow: text in the tool message, images in a user message."""
    tok = _tok()
    engine = ScriptedEngine(tok, tok.eos_token_id, [CALL, "Done."])
    options = {
        "env.image_results": "user",
        "env.image_tool_note": "[media attached in following user message]",
        "env.image_user_note": "(attached media from previous tool call)",
    }
    (group,) = _run(_pipeline(tok, engine, lambda: ScreenEnv(text_only=True), **options), [PROMPT])
    t = group[0]
    assert t.finish == "stop", t.info.get("error")
    assert [m["role"] for m in t.messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert t.messages[2]["content"] == "[media attached in following user message]"
    follow_up = t.messages[3]["content"]
    assert follow_up[0]["text"] == "(attached media from previous tool call)" and follow_up[1]["type"] == "image"
    context = tok.decode([x for x, m in zip(t.tokens, t.mask) if not m])
    assert "[media attached in following user message]\n</tool_response><|im_end|>" in context
    assert (
        "<|im_start|>user\n(attached media from previous tool call)<|vision_start|><|image_pad|><|vision_end|><|im_end|>"
        in context
    )


def test_an_episode_can_set_its_own_layout():
    """A harness profile's layout reaches the episode through episode_info (HarborEnv sets it from
    HarnessProfile.image_results) and wins over env.image_results."""

    class ChatCompletionsHarness(ScreenEnv):
        def reset(self, **row):
            super().reset(**row)
            self.episode_info = {"image_results": "user"}

    tok = _tok()
    engine = ScriptedEngine(tok, tok.eos_token_id, [CALL, "Done."])
    (group,) = _run(_pipeline(tok, engine, ChatCompletionsHarness), [PROMPT])
    assert [m["role"] for m in group[0].messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert group[0].messages[2]["content"] == "screen 1"

    from palingenesis.rl.envs.harness.profile import HarnessProfile

    assert HarnessProfile("custom", "1", image_results="user").image_results == "user"


def test_image_prompt_and_text_only_policy():
    tok = _tok()
    row = {
        "prompt": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is this?"}, {"type": "image", "image": tiny_image(16, 16)}],
            }
        ]
    }
    engine = ScriptedEngine(tok, tok.eos_token_id, ["A picture."])
    (group,) = _run(_pipeline(tok, engine, ScreenEnv), [row])
    assert group[0].finish == "stop" and len(group[0].images) == 1
    assert multimodal.vision_of(tok).count_images(group[0].prompt_ids) == 1
    # a text-only policy: images in the prompt are an error, images from tools are dropped
    text_tok = _tok(vision=False)
    engine = ScriptedEngine(text_tok, text_tok.eos_token_id, ["A picture."])
    (group,) = _run(_pipeline(text_tok, engine, ScreenEnv), [row])
    assert group[0].finish == "error" and "text-only" in group[0].info["error"]
    engine = ScriptedEngine(text_tok, text_tok.eos_token_id, [CALL, "Done."])
    (group,) = _run(_pipeline(text_tok, engine, ScreenEnv), [PROMPT])
    assert group[0].finish == "stop" and group[0].messages[2]["content"] == "screen 1" and not group[0].images


def test_template_that_drops_tool_images_is_reported():
    tok = _tok()
    # a template whose tool turns render text only (as most do): the image has nowhere to go
    tok.chat_template = tok.chat_template.replace(
        '{%- elif message.role == "tool" %}', "{%- elif message.role == \"tool\" %}{%- set content = 'text only' %}"
    )
    assert "text only" in tok.chat_template
    engine = ScriptedEngine(tok, tok.eos_token_id, [CALL, "Done."])
    (group,) = _run(_pipeline(tok, engine, ScreenEnv), [PROMPT])
    assert group[0].finish == "error" and 'env.image_results: "user"' in group[0].info["error"]
    engine = ScriptedEngine(tok, tok.eos_token_id, [CALL, "Done."])
    (group,) = _run(_pipeline(tok, engine, ScreenEnv, **{"env.image_results": "user"}), [PROMPT])
    assert group[0].finish == "stop" and len(group[0].images) == 1


def test_a_sampled_image_placeholder_is_an_error():
    tok = _tok()
    engine = ScriptedEngine(tok, tok.eos_token_id, ["Here <|image_pad|> it is."])
    (group,) = _run(_pipeline(tok, engine, ScreenEnv), [PROMPT])
    assert group[0].finish == "error" and not group[0].trained


# ----------------------------------------------------------------- rollout == training


def _trainer(model, tok, config=None):
    from palingenesis.logits import output_head
    from palingenesis.rl.losses import LowPrecisionWeight
    from palingenesis.rl.parallel import Parallel
    from palingenesis.rl.trainer import RLTrainer

    trainer = object.__new__(RLTrainer)
    trainer.config = config or RLConfig()
    trainer.config.set("train.prompt_sharing", "off")
    trainer.parallel, trainer.device, trainer.pad_id = Parallel(fsdp=False), "cpu", tok.pad_token_id
    trainer.model, trainer.head, trainer.head_weight = model, output_head(model), LowPrecisionWeight()
    trainer.inv_temperature = 1.0
    trainer.vision = multimodal.vision_of(tok)
    return trainer


class _ShortTurns(HFRollout):
    """The trainer's own HF engine, with every turn ending after its token budget as if on a stop
    token: a random policy then plays several turns (the pipeline closes the turn itself)."""

    def generate(self, prompts, max_new_tokens, temperature, images=None):
        rollouts = super().generate(prompts, max_new_tokens, temperature, images=images)
        for rollout in rollouts:
            rollout.finish_reason = "stop"
        return rollouts


def _episodes(tok, model, env, rows, size=3, **options):
    def engine(eot):
        return _ShortTurns(model, (eot,), tok.pad_token_id, 8, vision=multimodal.vision_of(tok))

    options = {"rollout.max_new_tokens": 6, **options}  # short turns: a random policy rarely stops by itself
    return _run(_pipeline(tok, engine, env, **options), rows, 1.0, size)


def _score(groups):
    for g, group in enumerate(groups):
        for i, t in enumerate(group):
            t.group, t.reward, t.scored, t.trained = g, float(i), True, True


@pytest.mark.parametrize("pruned", [False, True], ids=["all-images", "last-image-only"])
def test_training_scores_exactly_what_the_rollout_sampled(pruned):
    """On-policy, the trainer's log-probs of the sampled tokens equal the sampler's (ratio 1): same
    tokens after expansion, same M-RoPE positions, same pixels, turn after turn with new screenshots.
    With old screenshots dropped, every turn is still scored on the context it was sampled from."""
    torch.manual_seed(0)
    tok = _tok()
    model = tiny_qwen3_5(tok)
    options = {"env.max_images": 1} if pruned else {}
    rows = [PROMPT, {"prompt": "Close the window."}]
    groups = _episodes(tok, model, ActingEnv, rows, **options)
    for group in groups:
        for t in group:
            assert t.finish == "turns", t.info.get("error")
            assert t.turns >= 3 and t.sampled_tokens > 0
            if pruned:
                assert len(t.images) == 1 and len(t.segments) >= 2
                assert tok.decode(t.prompt_ids).count("[image omitted]") == len(t.segments)
                assert "<|vision_start|>[image omitted]" not in tok.decode(t.prompt_ids)  # the wrapper went too
            else:
                assert len(t.images) == t.turns - 1 and not t.segments
    _score(groups)
    stats = _trainer(model, tok)._train_step(groups)
    assert stats["trained_tokens"] == sum(t.sampled_tokens for g in groups for t in g)
    exact = stats["policy/abs_ratio_dev"]
    assert exact < 2e-5, exact
    # the pixels matter: with other images, the same tokens score differently
    for group in groups:
        for t in group:
            t.images = [tiny_image(im.height, im.width, 99) for im in t.images]
            for segment in t.segments:
                segment.images = [tiny_image(im.height, im.width, 99) for im in segment.images]
    model.zero_grad()
    assert _trainer(model, tok)._train_step(groups)["policy/abs_ratio_dev"] > 10 * max(exact, 1e-5)


def test_context_rewrite_keeps_every_turn_with_its_context():
    tok = _tok()
    vision = multimodal.vision_of(tok)
    engine = ScriptedEngine(tok, tok.eos_token_id, [CALL, CALL, CALL, "Done."])
    options = {"env.max_images": 1, "env.image_prune_slack": 1, "env.image_placeholder": "[screenshot removed]"}
    (group,) = _run(_pipeline(tok, engine, ScreenEnv, **options), [PROMPT])
    t = group[0]
    assert t.finish == "stop" and t.turns == 4, t.info.get("error")
    # 3 screenshots, keep 1 with a slack of 1: one rewrite, when the third arrives
    assert len(t.segments) == 1 and t.info["context_rewrites"] == 1
    assert [len(images) for _, images in engine.seen] == [0, 1, 2, 1]
    views = t.training_views()
    assert [len(v.images) for v in views] == [2, 1]
    assert sum(sum(v.mask) for v in views) == t.sampled_tokens
    for view, (prompt, images) in zip(views, (engine.seen[2], engine.seen[3])):
        # each view's sequence is the engine's own prompt for its last turn, then that turn
        assert (view.prompt_ids + view.tokens)[: len(prompt)] == prompt
        assert vision.count_images(view.prompt_ids + view.tokens) == len(view.images) == len(images)
    last = tok.decode(views[1].prompt_ids)
    assert last.count("[screenshot removed]") == 2 and last.count("<|image_pad|>") == 1
    assert "screen 1[screenshot removed]\n</tool_response>" in last  # the tool text stays
    assert views[0].advantage == t.advantage and views[0].group == t.group


def test_drop_images_edits_tokens_in_place():
    tok = _tok()
    vision = multimodal.vision_of(tok)
    chat = ChatFormat(tok, tok.eos_token_id, KWARGS)
    start, end = tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>")
    assert chat.image_wrapper(vision.image_token_id) == ([start], [end])
    ids = [1, start, vision.image_token_id, end, 2, start, vision.image_token_id, end, 3]
    note = tok.encode("[image omitted]", add_special_tokens=False)
    assert chat.drop_images(ids, vision.image_token_id, 1, "[image omitted]") == [1, *note, 2, *ids[5:]]
    assert chat.drop_images(ids, vision.image_token_id, 2, "[image omitted]") == ids
    assert chat.drop_images(ids, vision.image_token_id, 0, "") == [1, 2, 3]


# ----------------------------------------------------------------- generation client and engines


def test_same_tokens_with_other_images_are_separate_requests():
    import asyncio

    from palingenesis.rl.generation import GenerationClient

    class Streaming:
        version = 0

        def __init__(self):
            self.added, self.done = [], []

        def stream_add(self, request_id, prompt, max_new_tokens, temperature, n, images=None):
            self.added.append((n, images))
            self.done.append((request_id, [Rollout([7], [-0.1], "stop", 0) for _ in range(n)]))

        def stream_pending(self):
            return bool(self.done)

        def stream_step(self):
            done, self.done = self.done, []
            return done

    engine = Streaming()
    client = GenerationClient(engine, merge=True)
    a, b = tiny_image(8, 8, 0), tiny_image(8, 8, 1)

    async def main():
        return await asyncio.gather(
            client.generate([1, 2], 4, 1.0, [a]),
            client.generate([1, 2], 4, 1.0, [a]),
            client.generate([1, 2], 4, 1.0, [b]),
            client.generate([1, 2], 4, 1.0),
        )

    try:
        out = asyncio.run(main())
    finally:
        client.close()
    assert len(out) == 4
    added = {(n, 0 if images is None else id(images[0])) for n, images in engine.added}
    assert added == {(2, id(a)), (1, id(b)), (1, 0)} and len(engine.added) == 3


def test_vllm_requests_carry_unexpanded_tokens_and_images():
    from palingenesis.opd.rollout import VLLMColocateRollout

    image = tiny_image(8, 8)
    request = VLLMColocateRollout._request([1, 2, 3], [image])
    assert request["prompt_token_ids"] == [1, 2, 3] and request["multi_modal_data"]["image"][0].size == (8, 8)
    assert VLLMColocateRollout._request([1, 2, 3]) == {"prompt_token_ids": [1, 2, 3]}


def test_hf_engine_refuses_images_without_a_vision_processor():
    tok = _tok()
    engine = HFRollout(tiny_qwen3_5(tok), (tok.eos_token_id,), tok.pad_token_id, 4)
    with pytest.raises(ValueError, match="text-only rollout engine"):
        engine.generate([[1, multimodal.vision_of(tok).image_token_id]], [2], 1.0, images=[[tiny_image(8, 8)]])


# ----------------------------------------------------------------- config and trainer


def _config(**options):
    config = RLConfig()
    for key, value in {"model.policy": "x", "data.dataset": "rows.jsonl", "rewards.m.fn": "math", **options}.items():
        config.set(key, value)
    return config


@pytest.mark.parametrize(
    "options, match",
    [
        ({"model.multimodal": True, "rollout.backend": "vllm_server"}, "vllm_server"),
        ({"model.multimodal": True, "loss.kl_ref": "start"}, "kl_ref"),
        ({"model.multimodal": True, "train.prompt_sharing": "on"}, "prompt_sharing"),
        ({"env.image_results": "assistant"}, "image_results"),
        ({"env.max_images": 2}, "need model.multimodal"),
        ({"model.train_projector": True}, "need model.multimodal"),
    ],
)
def test_unsupported_settings_are_refused(options, match):
    with pytest.raises(RLConfigError, match=match):
        _config(**options).validate()


def test_rl_trainer_end_to_end(tmp_path):
    """RLTrainer on a multimodal policy with screenshots from the environment: trains the language
    model, leaves the vision tower as it was, exports the multimodal layout with its processor."""
    from safetensors.torch import load_file

    from palingenesis.rl.trainer import RLTrainer

    policy = save_tiny(tmp_path / "policy")
    rows = [{"prompt": f"Task {i}: open window {i}."} for i in range(12)]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def length(completion):  # dense, so a random policy's groups disagree
        return len(completion) / 40

    config = RLConfig()
    for key, value in {
        "model.policy": policy,
        "model.multimodal": True,
        "model.use_liger_kernel": False,
        "model.match_vllm_norm": False,
        "model.chat_template_kwargs": KWARGS,
        "model.gradient_checkpointing": "none",
        "data.dataset": str(tmp_path / "data.jsonl"),
        "data.eval_size": 2,
        "rollout.backend": "hf",
        "rollout.batch_prompts": 2,
        "rollout.group_size": 3,
        "rollout.max_new_tokens": 8,
        "rollout.max_model_len": 1024,
        "env.max_turns": 3,
        "env.max_images": 1,
        "train.output_dir": str(tmp_path / "run"),
        "train.steps": 2,
        "train.learning_rate": 1e-3,
        "train.warmup_steps": 1,
    }.items():
        config.set(key, value)
    trainer = RLTrainer(config, rewards=[length], env=ActingEnv)
    trainer.train()
    before = load_file(str(Path(policy) / "model.safetensors"))
    after = load_file(str(tmp_path / "run" / "final" / "model.safetensors"))
    assert set(before) == set(after)
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed and not any(".visual." in k for k in changed)
    assert (tmp_path / "run" / "final" / "preprocessor_config.json").exists()
