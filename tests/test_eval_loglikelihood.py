"""pgs eval, multiple choice by log-likelihood: the choice trie over next-token queries against direct sums of a
toy language model and of a small transformer, the anchor arithmetic for tokens outside the top candidates, the
request counts, lm-eval's pair tokenisation, and a choices task end to end (no GPU)."""

import asyncio
import json
import math

import pytest

torch = pytest.importorskip("torch")

from palingenesis.eval import loglikelihood as ll  # noqa: E402
from palingenesis.opd.rollout import Candidates, Rollout, Sampling, samplings  # noqa: E402


class ToyLM:
    """A deterministic next-token model served like an engine: generate() honours Sampling.allowed (the log-probs
    are renormalised over the allowed tokens) and Sampling.candidates (the top candidates), as vLLM's processed
    log-probs do. logits(prefix) -> a list of `vocab` floats."""

    def __init__(self, vocab=40, logits=None):
        self.vocab, self.version, self.requests = vocab, 0, []
        self._logits = logits or self._hashed

    def _hashed(self, prefix):
        g = torch.Generator().manual_seed(hash(tuple(prefix[-3:])) % (2**31))
        return (torch.randn(self.vocab, generator=g) * 3).tolist()

    def logprobs(self, prefix):
        return torch.log_softmax(torch.tensor(self._logits(list(prefix)), dtype=torch.float64), -1)

    def loglikelihood(self, context, continuation):
        total, prefix = 0.0, list(context)
        for token in continuation:
            total += float(self.logprobs(prefix)[token])
            prefix.append(token)
        return total

    def generate(self, prompts, max_new_tokens, temperature):
        out = []
        for prompt, budget, sampling in zip(prompts, max_new_tokens, samplings(temperature, len(prompts))):
            assert budget == 1 and sampling.temperature == 1.0
            self.requests.append((tuple(prompt), sampling.allowed))
            logprobs = self.logprobs(prompt)
            if sampling.allowed:
                mask = torch.full_like(logprobs, float("-inf"))
                mask[list(sampling.allowed)] = 0
                logprobs = torch.log_softmax(logprobs + mask, -1)
            top = torch.topk(logprobs, min(sampling.candidates, self.vocab))
            token = int(torch.multinomial(logprobs.exp(), 1))
            candidates = Candidates.from_rows([(top.indices.tolist(), top.values.tolist())])
            out.append(Rollout([token], [float(logprobs[token])], "length", 0, candidates=candidates))
        return out

    def update_weights(self, named, version):
        self.version = version

    def sleep(self):
        pass

    def wake(self):
        pass


def _score(engine, context, choices, top):
    from palingenesis.rl.generation import GenerationClient

    client = GenerationClient(engine, merge=True)

    async def run():
        client.enter()
        try:
            return await ll.trie_loglikelihoods(ll.client_peek(client, top), context, choices, top)
        finally:
            client.exit()

    try:
        return asyncio.run(run())
    finally:
        client.close()


def test_trie_scores_equal_direct_sums_whatever_the_rank():
    lm = ToyLM()
    context = [3, 1, 4, 1, 5]
    # single tokens, a shared first token, a shared two-token prefix, a choice that is a prefix of another
    choices = [[7], [8], [9, 2], [9, 3], [11, 12, 13], [11, 12, 14, 15], [11, 12]]
    for top in (2, 4, 40):  # 2: almost every wanted token is outside the candidates and needs the anchor request
        lm.requests.clear()
        scores = _score(lm, context, choices, top)
        expected = [lm.loglikelihood(context, c) for c in choices]
        assert scores.loglikelihoods == pytest.approx(expected, abs=1e-4), top
        assert scores.requests == len(lm.requests)
        for choice, greedy in zip(choices, scores.greedy):
            prefix, argmax = list(context), True
            for token in choice:
                argmax = argmax and int(lm.logprobs(prefix).argmax()) == token
                prefix.append(token)
            assert greedy == argmax
    # with every candidate returned, a node is one request: 5 trie nodes branch or continue here (7 choices,
    # 14 choice tokens; one request per choice would prefill the context 7 times)
    assert scores.requests == 5
    assert {r[0][len(context) :] for r in lm.requests} == {(), (9,), (11,), (11, 12), (11, 12, 14)}


def test_single_token_choices_cost_one_request_and_the_root_goes_first():
    lm = ToyLM()
    context = list(range(10))
    scores = _score(lm, context, [[1], [2], [3], [4]], top=40)
    assert scores.requests == 1 and lm.requests == [(tuple(context), ())]
    # outside the top: one more request, restricted to the anchor (the most likely token) and the missing tokens
    lm.requests.clear()
    ranked = lm.logprobs(context).argsort(descending=True).tolist()
    rare = ranked[-3:]
    scores = _score(lm, context, [[t] for t in rare], top=8)
    assert scores.requests == 2 and lm.requests[1][1] == (ranked[0], *sorted(rare))
    assert scores.loglikelihoods == pytest.approx([float(lm.logprobs(context)[t]) for t in rare], abs=1e-4)
    # more missing tokens than a request can report: chunks of top - 1, each with the anchor
    lm.requests.clear()
    many = ranked[10:30]
    scores = _score(lm, context, [[t] for t in many], top=8)
    assert scores.requests == 1 + math.ceil(len(many) / 7)
    assert scores.loglikelihoods == pytest.approx([float(lm.logprobs(context)[t]) for t in many], abs=1e-4)
    # deeper nodes wait for the root, whose prefill fills the prefix cache
    lm.requests.clear()
    _score(lm, context, [[5, 6], [5, 7], [8, 9]], top=40)
    assert lm.requests[0][0] == tuple(context) and len(lm.requests) == 3
    # an empty continuation has log-probability 0 and asks nothing
    assert _score(lm, context, [[], [1]], top=40).loglikelihoods[0] == 0.0


def test_impossible_tokens_score_minus_infinity():
    def logits(prefix):
        row = [0.0] * 12
        row[5] = float("-inf")
        return row

    scores = _score(ToyLM(12, logits), [1, 2], [[5], [6]], top=4)
    assert scores.loglikelihoods[0] == float("-inf") and scores.loglikelihoods[1] == pytest.approx(math.log(1 / 11))
    metrics = ll.choice_metrics(scores, ["a", "b"], gold=1)
    assert metrics["acc"] == 1.0 and metrics["prob_gold"] == pytest.approx(1.0) and metrics["loglikelihoods"][0] < -1e29


def _tiny_model():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    return GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=64, n_embd=32, n_layer=2, n_head=2)).eval()


def test_trie_matches_the_forward_pass_of_a_transformer():
    model = _tiny_model()

    def logits(prefix):
        with torch.no_grad():
            return model(input_ids=torch.tensor([prefix])).logits[0, -1].tolist()

    context = [5, 9, 2, 33, 7, 1]
    choices = [[10], [11], [12, 13, 14], [12, 13, 15], [40, 41]]
    pairs = [(context, c) for c in choices]
    forward = ll.forward_loglikelihoods(model, pad_id=0, pairs=pairs, batch=2)
    with torch.no_grad():
        for (lp, greedy), choice in zip(forward, choices):  # the padded batch equals each sequence alone
            alone = torch.log_softmax(model(input_ids=torch.tensor([context + choice])).logits[0].double(), -1)
            steps = [alone[len(context) - 1 + i, t] for i, t in enumerate(choice)]
            assert lp == pytest.approx(float(sum(steps)), abs=1e-4)
            assert greedy == all(int(alone[len(context) - 1 + i].argmax()) == t for i, t in enumerate(choice))
    for top in (3, 64):
        trie = _score(ToyLM(64, logits), context, choices, top)
        assert trie.loglikelihoods == pytest.approx([f[0] for f in forward], abs=1e-3)
        assert trie.greedy == [f[1] for f in forward]


def test_choice_metrics_follow_lm_eval():
    scores = ll.ChoiceScores([-4.0, -3.0, -3.5], [False, True, False])
    m = ll.choice_metrics(scores, ["a", "a much longer choice", "bb"], gold=1)
    assert (m["acc"], m["pred"]) == (1.0, 1) and m["acc_norm"] == 1.0  # -3 / 20 per character is the best
    m = ll.choice_metrics(ll.ChoiceScores([-1.0, -2.0], [True, False]), ["yes", "no, because it is long"], gold=0)
    assert m["acc"] == 1.0 and m["acc_norm"] == 0.0  # length normalisation prefers the long choice
    assert m["prob_gold"] == pytest.approx(math.exp(-1) / (math.exp(-1) + math.exp(-2)))


def test_encode_pair_moves_trailing_spaces_to_the_continuation():
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    context, continuation = ll.encode_pair(tok, "Question: 2 + 2?\nAnswer: ", "4")
    assert tok.decode(context) == "Question: 2 + 2?\nAnswer:" and tok.decode(continuation) == " 4"
    assert context + continuation == tok.encode("Question: 2 + 2?\nAnswer: 4", add_special_tokens=False)
    contexts = ["Q: one\nA:", "Q: two words here\nA: "]
    choices = [[" A", " B", " C"], ["yes", "no"]]
    batched = ll.encode_pairs(tok, contexts, choices)
    for context, options, (ids, encoded) in zip(contexts, choices, batched):
        for option, got in zip(options, encoded):
            assert (ids, got) == ll.encode_pair(tok, context, option)
    assert ll.encode_pairs(tok, [], []) == []


def test_sampling_carries_allowed_tokens_and_stop_strings():
    s = Sampling(1.0, candidates=8, allowed=[3, 1], stop=["\n\n"])
    assert s.allowed == (3, 1) and s.stop == ("\n\n",) and hash(s) == hash(Sampling(1.0, candidates=8, allowed=(3, 1), stop=("\n\n",)))
    import dataclasses

    assert Sampling(**json.loads(json.dumps(dataclasses.asdict(s)))) == s  # the engine pool's wire format
    from palingenesis.opd.sglang_rollout import sampling_params

    assert sampling_params(Sampling(0.7, stop=("Question:",)), 16, [2])["stop"] == ["Question:"]
    with pytest.raises(NotImplementedError, match="Sampling.allowed"):
        sampling_params(s, 16, [2])


# ------------------------------------------------------------ a task, end to end


@pytest.fixture
def qwen(monkeypatch):
    from palingenesis.eval.config import SamplingConfig
    from palingenesis.opd.teachers import end_of_turn_id
    from tests.test_rl import tokenizer

    tok = tokenizer("Qwen/Qwen3-0.6B")
    eot = end_of_turn_id(tok, {"enable_thinking": False})
    monkeypatch.setattr("palingenesis.opd.trainer.student_stop_ids", lambda *a, **k: (eot,))
    monkeypatch.setattr("palingenesis.eval.runner.model_sampling", lambda model: SamplingConfig())
    return tok


class LetterLM(ToyLM):
    """Prefers the token of the letter written after "KEY=" in the prompt; a little noise otherwise."""

    def __init__(self, tok):
        super().__init__(vocab=len(tok))
        self.tok = tok

    def _hashed(self, prefix):
        g = torch.Generator().manual_seed(hash(tuple(prefix[-4:])) % (2**31))
        row = torch.randn(self.vocab, generator=g)
        text = self.tok.decode(prefix)
        if "KEY=" in text:
            letter = text.split("KEY=")[-1][0]
            row[self.tok.encode(" " + letter, add_special_tokens=False)[0]] += 12.0
        return row.tolist()


def test_choices_task_end_to_end(tmp_path, qwen):
    from palingenesis.eval.config import load_task
    from palingenesis.eval.runner import Evaluator

    tok = qwen
    rows = [
        {"q": "KEY=B first", "options": ["A", "B", "C", "D"], "label": 1, "topic": "x"},
        {"q": "KEY=D second", "options": ["A", "B", "C", "D"], "label": "A", "topic": "x"},  # the model says D
        {"q": "KEY=C third", "options": ["A", "B", "C"], "label": 2, "topic": "y"},
        {"q": "KEY=A fourth", "options": ["Alpha centauri", "A", "An apple a day"], "label": 1, "topic": "y"},
    ]
    data = tmp_path / "mc.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    task = load_task(
        {
            "name": "letters", "dataset": str(data), "output": "choices", "choices_field": "options",
            "gold_field": "label", "prompt": "{{q}}\nAnswer:", "template": "completion", "category_field": "topic",
        }
    )  # fmt: skip
    engine = LetterLM(tok)
    out = tmp_path / "out"
    with Evaluator(tok, engine, "", engine_options={"max_model_len": 512}, resamples=200, logprob_top=20) as evaluator:
        summary = evaluator.run(task, out_dir=out)
    assert summary["items"] == 4 and summary["mean_score"]["estimate"] == pytest.approx(0.75)
    assert summary["output"] == "choices" and summary["k"] == 1 and summary["generated_tokens"] == summary["engine_requests"]
    assert summary["categories"] == {
        "x": {"items": 2, "score": 0.5, "stderr": pytest.approx(0.5)},
        "y": {"items": 2, "score": 1.0, "stderr": 0.0},
    }
    assert set(summary["metrics"]) == {"acc", "acc_norm"} and summary["metrics"]["acc"]["mean"] == pytest.approx(0.75)
    result = [json.loads(line) for line in (out / "letters.default.jsonl").read_text().splitlines()]
    assert [r["pred"] for r in result] == [1, 3, 2, 1] and [r["gold"] for r in result] == [1, 0, 2, 1]
    # every log-likelihood is the model's own: the direct sum over the choice's tokens after the prompt
    samples = [json.loads(line) for line in (out / "letters.default.samples.jsonl").read_text().splitlines()]
    by_id = {s["id"]: s for s in samples}
    for index, row in enumerate(rows):
        context = tok.encode(f"{row['q']}\nAnswer:", add_special_tokens=False)
        for option, got in zip(row["options"], by_id[f"letters/{index}"]["loglikelihoods"]):
            continuation = tok.encode(f"{row['q']}\nAnswer: {option}", add_special_tokens=False)[len(context) :]
            assert got == pytest.approx(engine.loglikelihood(context, continuation), abs=2e-3)
    # the three single-letter items took one request each (the letters are among the top candidates or one more)
    assert all(by_id[f"letters/{i}"]["requests"] <= 2 for i in range(3))


def test_choices_with_the_model_in_process_use_forward_passes(tmp_path, qwen):
    """An engine that holds its model (the hf backend) is scored with padded forwards, not through the trie."""
    from palingenesis.eval.config import load_task
    from palingenesis.eval.runner import Evaluator

    tok = qwen
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=16, n_layer=1, n_head=2)).eval()

    class Engine:
        version, pad_id, micro_seqs = 0, 0, 2

        def __init__(self):
            self.model = model

        def generate(self, *a):
            raise AssertionError("no sampling request is needed")

    rows = [{"q": f"item {i}", "choices": ["yes", "no", "maybe so"], "gold": i % 3} for i in range(5)]
    data = tmp_path / "mc.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    task = load_task({"name": "mc", "dataset": str(data), "output": "choices", "prompt": "{{q}}?", "template": "completion"})
    with Evaluator(tok, Engine(), "", engine_options={"max_model_len": 64}, resamples=100) as evaluator:
        summary = evaluator.run(task, out_dir=tmp_path / "out")
    assert summary["items"] == 5 and summary["engine_requests"] == 15
    samples = [json.loads(line) for line in (tmp_path / "out" / "mc.default.samples.jsonl").read_text().splitlines()]
    pairs = ll.encode_pairs(tok, ["item 0?"], [[" yes", " no", " maybe so"]])[0]
    expected = ll.forward_loglikelihoods(model, 0, [(pairs[0], c) for c in pairs[1]])
    first = next(s for s in samples if s["id"] == "mc/0")
    assert first["loglikelihoods"] == pytest.approx([e[0] for e in expected], abs=1e-3)
