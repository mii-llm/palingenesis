"""Serving: publish a scope's stacked adapter to inference servers under versioned names, check it with a
parity canary, roll back, and unload old versions.

    name        <scope>@v<n>, e.g. user:acme/support/ana@v3, where n counts the scope's publishes. Every
                publish is a new name (a new LoRA id), never an in-place reload: vLLM keys its prefix cache by
                adapter name, so KV blocks (and hybrid recurrent-state checkpoints) computed under an old
                version are never reused, and in-flight requests finish on the version they started with
    composite   the chain's live adapters (org + team + user) stacked exactly into one PEFT LoRA of the summed
                rank, padded to a rank vLLM accepts (Learner.stack); SGLang gets a copy with its module prefix
    publish     1. refuse a chain with a forced or ungated version, a failed evaluation, or an evaluation on
                   other parent versions than the live ones, unless forced (recorded with the reasons)
                2. load the composite on every server (vLLM /v1/load_lora_adapter, SGLang /load_lora_adapter,
                   or the vLLM resolver plugin on the first request) and warm it with a one-token request
                3. run the parity canary on every server; if any server fails, the new name is unloaded
                   everywhere and the old version stays live
                4. flip: published.json records the new live version (the proxy routes to it); the old one
                   drains for serving.drain_seconds, then `pgs learn unload`, the next publish or the next sleep
                   unloads it
    canary      acquisition probes of the chain, answered greedily by the HF + PEFT reference loaded with the
                same composite (preferring probes whose answer the adapter changes). On each server: the
                answers are right as often as HF's (paired counts); the adapter's per-token effect (logprob
                with the adapter minus without, on HF's answer tokens) correlates with HF's; and the k3 mismatch
                (rl.losses.mismatch_by_bucket) with the adapter stays within a ratio of the base model's own
                HF-vs-server k3 on the same tokens, so hybrid numerics are not blamed on the export. A server
                that ignores the adapter, a wrong module prefix or a dropped module fails it
    rollback    flips back to an earlier version (loaded again if it was unloaded); the scope's LIVE follows,
                and what the rolled-back versions consolidated counts as unconsolidated again (back in the
                served prompt, trained again by the next session)

Runtime LoRA updates: vLLM exposes the load / unload routes, and calls resolver plugins, only with
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True and --enable-lora; vLLM warns that this belongs on trusted, isolated
networks only. SGLang needs --enable-lora; its load / unload routes share the inference port, so run it with
--api-key on a private network. The proxy never forwards these routes.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from palingenesis.learn.store import Note, ScopeId, Store, StoreError, now

logger = logging.getLogger(__name__)
_VERSION = re.compile(r"^v([1-9][0-9]*)$")
LIVE_STATES = ("live", "draining", "retired")


class PublishError(StoreError):
    """A publish, rollback or unload that cannot proceed; the live version is unchanged."""


class RuntimeLoRAError(PublishError):
    """The server does not accept LoRA adapters at runtime (missing flag or environment variable)."""


class CanaryError(PublishError):
    """The parity canary failed on a server: the new version was unloaded, the old one stays live."""


# ---------------------------------------------------------------------------- names and records


def served_name(scope: str | ScopeId, n: int) -> str:
    return f"{ScopeId.parse(scope)}@v{n}"


def parse_name(name: str) -> tuple[ScopeId, int | None]:
    """(scope, n) of `<scope>@v<n>`, or (scope, None) for the bare `<scope>` (the live version). StoreError
    when it is neither."""
    scope, sep, version = str(name).rpartition("@")
    if not sep:
        return ScopeId.parse(name), None
    m = _VERSION.match(version)
    if m is None:
        raise StoreError(f"bad served name {name!r}: use <scope>@v<n> or <scope>")
    return ScopeId.parse(scope), int(m.group(1))


def same_model(a: str, b: str) -> bool:
    """Two names of the same base model: equal, or the same last path component (a hub id and a local copy)."""
    a, b = str(a).rstrip("/"), str(b).rstrip("/")
    return a == b or (bool(a) and a.rsplit("/", 1)[-1] == b.rsplit("/", 1)[-1])


def scope_record(published: dict, scope: str | ScopeId) -> dict:
    return published["scopes"].setdefault(
        str(ScopeId.parse(scope)), {"live": None, "next": 1, "versions": {}, "history": []}
    )


def served_version(store: Store, scope: str | ScopeId, published: dict | None = None) -> tuple[str | None, dict | None]:
    """(scope it belongs to, version entry) of what a request for `scope` is served with: the scope's live
    published version, else its nearest ancestor's (a user without an adapter of its own is served its team's
    or org's composite); (None, None) when nothing in the chain is published."""
    published = store.published() if published is None else published
    for s in reversed(ScopeId.parse(scope).chain()):
        rec = published["scopes"].get(str(s))
        if rec and rec.get("live") is not None:
            return str(s), rec["versions"][str(rec["live"])]
    return None, None


def wake_notes(store: Store, scope: str | ScopeId, entry: dict | None) -> list[Note]:
    """The chain's active notes that the served composite `entry` does not hold (ancestors first, oldest
    first): the notes a served prompt carries until a published version consolidates them. Rolling back
    re-injects what the rolled-back version had consolidated."""
    consolidated: dict[str, set] = {}
    for s, version in (entry or {}).get("chain", []):
        manifest = store.scope(s).manifest(version)
        consolidated[s] = set() if manifest.get("pruned") else set(manifest.get("note_ids", []))
    out = []
    for s in ScopeId.parse(scope).chain():
        ss = store.scope(s)
        if ss.exists:
            held = consolidated.get(str(s), set())
            out += sorted((n for n in ss.active_notes() if n.id not in held), key=lambda n: n.created)
    return out


def publish_check(store: Store, scope: str | ScopeId) -> tuple[list[tuple[str, str]], list[str]]:
    """(chain of (scope, live version), reasons it may not be published). Reasons: a chain version that went
    live forced or ungated, a failed evaluation, or an evaluation on other parent versions than the parents'
    live ones (the scope must be evaluated again on them). PublishError when nothing can be served."""
    from palingenesis.learn.evaluate import chain_versions, load_report

    scope = ScopeId.parse(scope)
    chain = []
    for s in scope.chain():
        ss = store.scope(s)
        if ss.exists and ss.live is not None:
            chain.append((str(s), ss.live))
    if not chain:
        raise PublishError(f"no live adapter in the chain of {scope}: train a version and set it live first")
    reasons = []
    for s, version in chain:
        ss = store.scope(s)
        manifest = ss.manifest(version)
        if manifest.get("pruned"):
            raise PublishError(f"{s} {version} was pruned ({manifest['pruned']['reason']}): rebuild the scope first")
        if manifest.get("forced"):
            reasons.append(f"{s} {version} went live forced past: {'; '.join(manifest.get('forced_reasons') or [])}")
        elif not manifest.get("gated"):
            reasons.append(f"{s} {version} went live without passing its gates")
        report = load_report(ss, version)
        if report is None:
            reasons.append(f"{s} {version} has no evaluation")
            continue
        failed = [g["name"] for g in report["gates"] if not g["passed"]]
        if failed and not manifest.get("forced"):
            reasons.append(f"{s} {version} failed gates: {', '.join(failed)}")
        parents = chain_versions(store, ScopeId.parse(s))[:-1]
        if report["live_chain"][:-1] != parents:
            reasons.append(
                f"{s} {version} was evaluated on parents {report['live_chain'][:-1]}, but their live versions are "
                f"{parents}: evaluate it again (pgs learn eval --scope {s} --version {version})"
            )
    return chain, reasons


# ---------------------------------------------------------------------------- server clients


class ServerClient:
    """An inference server's LoRA and scoring routes. `adapter` None means the base model."""

    engine = ""
    runtime_hint = ""

    def __init__(self, url: str, base_name: str, api_key: str = "", timeout: float = 120.0):
        self.url, self.base_name, self.api_key, self.timeout = url.rstrip("/"), base_name, api_key, timeout

    def request(self, method: str, path: str, payload: dict | None = None) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(f"{self.url}{path}", data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise PublishError(f"{self.url}: unreachable ({getattr(e, 'reason', e)})") from None

    def json(self, method: str, path: str, payload: dict | None = None) -> dict:
        status, body = self.request(method, path, payload)
        if status >= 400:
            raise PublishError(f"{self.url}{path}: HTTP {status}: {body.decode(errors='replace')[:500]}")
        return json.loads(body) if body else {}

    def model_ids(self) -> list[str]:
        cards = self.json("GET", "/v1/models").get("data", [])
        return [c.get(k) for c in cards for k in ("id", "root") if c.get(k)]

    def check_base(self) -> None:
        ids = self.model_ids()
        if not any(same_model(self.base_name, i) for i in ids):
            raise PublishError(
                f"{self.url} serves {sorted(set(ids))}, not the store's base {self.base_name!r}: adapters only fit "
                "the base they were trained on (set serving.base_name if the server names it differently)"
            )

    def _runtime_error(self, path: str) -> RuntimeLoRAError:
        return RuntimeLoRAError(f"{self.url} has no {path} route: {self.runtime_hint}")

    def load(self, name: str, path: str) -> None:
        raise NotImplementedError

    def unload(self, name: str) -> bool:
        """Unload `name`; False when the server did not have it."""
        raise NotImplementedError

    def generate(self, adapter: str | None, prompt: list[int] | str, max_tokens: int) -> str:
        raise NotImplementedError

    def prompt_logprobs(self, adapter: str | None, ids: list[int], start: int) -> list[float]:
        """Log-probabilities of ids[start:], each given the tokens before it."""
        raise NotImplementedError


class VLLMClient(ServerClient):
    engine = "vllm"
    runtime_hint = (
        "start vLLM with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True and --enable-lora --max-lora-rank R (vLLM allows "
        "this only on trusted, isolated networks), or publish with serving.transport resolver and the "
        "palingenesis_learn plugin (which needs the same variable)"
    )

    @staticmethod
    def _route_missing(status: int, body: bytes) -> bool:
        return status in (404, 405) and b"NotFoundError" not in body and b"cannot be found" not in body

    def load(self, name: str, path: str) -> None:
        status, body = self.request("POST", "/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
        if self._route_missing(status, body):
            raise self._runtime_error("/v1/load_lora_adapter")
        if status == 400 and b"already been loaded" in body:
            return
        if status >= 400:
            raise PublishError(f"{self.url}: loading {name}: HTTP {status}: {body.decode(errors='replace')[:500]}")

    def unload(self, name: str) -> bool:
        status, body = self.request("POST", "/v1/unload_lora_adapter", {"lora_name": name})
        if self._route_missing(status, body):
            raise self._runtime_error("/v1/unload_lora_adapter")
        if status == 404:
            return False
        if status >= 400:
            raise PublishError(f"{self.url}: unloading {name}: HTTP {status}: {body.decode(errors='replace')[:500]}")
        return True

    def _complete(self, adapter: str | None, prompt, **params) -> dict:
        payload = {"model": adapter or self.base_name, "prompt": prompt, "temperature": 0.0, **params}
        status, body = self.request("POST", "/v1/completions", payload)
        if status == 404 and adapter and b"does not exist" in body:
            raise RuntimeLoRAError(
                f"{self.url} does not know {adapter}: load it first, or start vLLM with VLLM_PLUGINS="
                "palingenesis_learn, VLLM_ALLOW_RUNTIME_LORA_UPDATING=True and PGS_LEARN_STORE set to the store"
            )
        if status >= 400:
            raise PublishError(f"{self.url}/v1/completions: HTTP {status}: {body.decode(errors='replace')[:500]}")
        return json.loads(body)["choices"][0]

    def generate(self, adapter, prompt, max_tokens):
        return self._complete(adapter, prompt, max_tokens=max_tokens)["text"]

    def prompt_logprobs(self, adapter, ids, start):
        rows = self._complete(adapter, list(ids), max_tokens=1, prompt_logprobs=0)["prompt_logprobs"]
        out = []
        for i in range(start, len(ids)):
            entry = rows[i] or {}
            value = entry.get(str(ids[i]), entry.get(ids[i]))
            if value is None:
                raise PublishError(f"{self.url}: no prompt logprob for token {ids[i]} at position {i}")
            out.append(float(value["logprob"] if isinstance(value, dict) else value))
        return out


class SGLangClient(ServerClient):
    engine = "sglang"
    runtime_hint = (
        "start SGLang with --enable-lora --max-lora-rank R (and --api-key on a private network: its LoRA routes "
        "share the inference port)"
    )

    def load(self, name: str, path: str) -> None:
        status, body = self.request("POST", "/load_lora_adapter", {"lora_name": name, "lora_path": path})
        if status in (404, 405):
            raise self._runtime_error("/load_lora_adapter")
        if status >= 400 or b'"success": false' in body or b'"success":false' in body:
            if b"already" in body:
                return
            raise PublishError(f"{self.url}: loading {name}: HTTP {status}: {body.decode(errors='replace')[:500]}")

    def unload(self, name: str) -> bool:
        status, body = self.request("POST", "/unload_lora_adapter", {"lora_name": name})
        if status in (404, 405) and not body.strip().startswith(b"{"):
            raise self._runtime_error("/unload_lora_adapter")
        return status < 400 and b'"success": false' not in body and b'"success":false' not in body

    def _generate(self, adapter, prompt, **extra) -> dict:
        payload = {"input_ids" if isinstance(prompt, list) else "text": prompt, **extra}
        if adapter:
            payload["lora_path"] = adapter
        return self.json("POST", "/generate", payload)

    def generate(self, adapter, prompt, max_tokens):
        return self._generate(adapter, prompt, sampling_params={"max_new_tokens": max_tokens, "temperature": 0.0})[
            "text"
        ]

    def prompt_logprobs(self, adapter, ids, start):
        out = self._generate(
            adapter,
            list(ids),
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            return_logprob=True,
            logprob_start_len=0,
        )
        rows = out["meta_info"]["input_token_logprobs"]
        return [float(r[0]) for r in rows[start : len(ids)]]


CLIENTS = {"vllm": VLLMClient, "sglang": SGLangClient}


def make_client(engine: str, url: str, base_name: str, api_key: str = "", timeout: float = 120.0) -> ServerClient:
    if engine not in CLIENTS:
        raise PublishError(f"unknown serving engine {engine!r} (vllm, sglang)")
    return CLIENTS[engine](url, base_name, api_key, timeout)


def engine_adapter(composite: Path, engine: str) -> Path:
    """The composite as the engine loads it: vLLM reads the PEFT names as written; SGLang drops the
    `language_model.` level of multimodal checkpoints (it renames the base weights the same way)."""
    composite = Path(composite)
    if engine == "vllm":
        return composite
    from palingenesis.learn import adapter as adapters

    out = composite / engine
    if not (out / adapters.ADAPTER_CONFIG).exists():
        adapters.rename_modules(composite, out, {"model.language_model.": "model."})
    return out


# ---------------------------------------------------------------------------- the parity canary


class Reference(Protocol):
    """The HF + PEFT side of the canary, holding the same composite as the servers."""

    def encode(self, messages: list[dict]) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def greedy(self, ids: list[int], max_new_tokens: int, adapter: bool) -> list[int]: ...

    def logprobs(self, ids: list[int], start: int, adapter: bool) -> list[float]: ...


class HFReference:
    """transformers + PEFT with the composite loaded; greedy answers and teacher-forced logprobs, with the
    adapter on or off (PEFT's disable_adapter)."""

    def __init__(
        self, base: str, composite: str | Path, device: str = "auto", chat_template_kwargs: dict | None = None
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from palingenesis.learn.adapter import require_peft
        from palingenesis.opd.trainer import student_stop_ids

        peft = require_peft()
        if device == "auto":
            from palingenesis.opd.teachers import pick_device

            device = pick_device()
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(base)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)
        self.model = peft.PeftModel.from_pretrained(model, str(composite)).eval()
        self.kwargs = chat_template_kwargs or {}
        self.stop = set(student_stop_ids(self.tok, base, [], self.kwargs))

    def _ctx(self, adapter: bool):
        import contextlib

        return contextlib.nullcontext() if adapter else self.model.disable_adapter()

    def encode(self, messages):
        from palingenesis.opd.formatting import encode_prompt

        return encode_prompt(self.tok, messages, self.kwargs)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def greedy(self, ids, max_new_tokens, adapter):
        torch = self.torch
        with torch.no_grad(), self._ctx(adapter):
            out = self.model.generate(
                input_ids=torch.tensor([ids], device=self.device),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=self.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else min(self.stop),
            )[0, len(ids) :].tolist()
        for i, t in enumerate(out):
            if t in self.stop:
                return out[: i + 1]
        return out

    def logprobs(self, ids, start, adapter):
        torch = self.torch
        with torch.no_grad(), self._ctx(adapter):
            logits = self.model(input_ids=torch.tensor([ids], device=self.device)).logits[0].float()
        lp = torch.log_softmax(logits[start - 1 : len(ids) - 1], dim=-1)
        target = torch.tensor(ids[start:], device=self.device)
        return lp.gather(1, target[:, None])[:, 0].tolist()


@dataclass
class CanaryResult:
    server: str
    passed: bool
    probes: int = 0
    tokens: int = 0
    reference_right: int = 0
    server_right: int = 0
    net_loss: int = 0  # probes HF answered right and the server wrong, minus the reverse
    effect_corr: float | None = None
    k3_adapter: float | None = None
    k3_base: float | None = None
    k3_ratio: float | None = None
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def line(self) -> str:
        def f(x):
            return "n/a" if x is None else f"{x:.4g}"

        verdict = "pass" if self.passed else "FAIL: " + "; ".join(self.reasons)
        return (
            f"{self.server}: {self.probes} probes, {self.tokens} tokens; right HF {self.reference_right} / server "
            f"{self.server_right}; effect corr {f(self.effect_corr)}; k3 {f(self.k3_adapter)} vs base "
            f"{f(self.k3_base)} (ratio {f(self.k3_ratio)}) {verdict}"
        )


def canary_probes(store: Store, chain: list, k: int, rng: random.Random) -> list:
    """Probes of the facts the chain's versions consolidated (every scope of the chain), with their gold values
    and competitors, at most k."""
    from palingenesis.learn.evaluate import Probe, _competitors, _resolve_all, _session_rows

    chain_facts = [f for s, _ in chain for f in store.scope(s).active_facts()]
    probes, seen = [], set()
    for s, _ in chain:
        ss = store.scope(s)
        facts = {f.id: f for f in ss.facts()}
        for row in _session_rows(ss, "probes.jsonl"):
            if row.get("kind") != "probe" or row["q"] in seen:
                continue
            resolved = _resolve_all(row.get("fact_ids", []), facts)
            if resolved is None or any(f.consolidated_in is None or f.status != "active" for f in resolved):
                continue
            seen.add(row["q"])
            competitors = sorted({c for f in resolved for c in _competitors(f, chain_facts)})
            stale = sorted({v for f in resolved for v in f.stale_values})
            probes.append(
                Probe("acquisition", row["q"], list(row["fact_ids"]), [f.value for f in resolved], competitors, stale)
            )
    return probes if len(probes) <= k else rng.sample(probes, k)


def _pearson(x: list[float], y: list[float]) -> tuple[float | None, float, float]:
    """(correlation or None, std of x, std of y)."""
    n = len(x)
    if n < 2:
        return None, 0.0, 0.0
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x) / n)
    sy = math.sqrt(sum((b - my) ** 2 for b in y) / n)
    if sx < 1e-6 or sy < 1e-6:
        return None, sx, sy
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (n * sx * sy), sx, sy


def k3_mismatch(reference: list[float], served: list[float], positions: list[int]) -> float | None:
    """Mean per-token k3 = rho - 1 - log rho, rho = exp(lp_reference - lp_served), over every position bucket of
    rl.losses.mismatch_by_bucket (the RL trainer's sampler-mismatch metric)."""
    if not reference:
        return None
    import torch

    from palingenesis.rl.losses import mismatch_by_bucket

    position = torch.tensor(positions, dtype=torch.long)
    stats = mismatch_by_bucket(
        torch.tensor(reference, dtype=torch.float64),
        torch.tensor(served, dtype=torch.float64),
        position,
        torch.full_like(position, -1),
    )
    buckets = ("pos_0_1k", "pos_1k_4k", "pos_4k_plus")
    tokens = sum(float(stats[f"bucket_tokens/{b}"]) for b in buckets)
    return sum(float(stats[f"mismatch_k3/{b}"]) for b in buckets) / tokens if tokens else None


def run_canary(
    probes: list,
    client: ServerClient,
    reference: Reference,
    name: str,
    gates,
    system_prompt: str = "",
    max_new_tokens: int = 48,
) -> CanaryResult:
    """The parity canary of `name` on one server against the HF + PEFT reference (thresholds: gates.parity_*)."""
    from palingenesis.learn.evaluate import exclusive_match

    result = CanaryResult(client.url, False)
    if not probes:
        result.reasons.append("no probes of consolidated facts to verify the adapter on (--no-canary skips, recorded)")
        return result
    rows = []
    for p in probes:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": "user", "content": p.q}
        ]
        ids = reference.encode(messages)
        on = reference.greedy(ids, max_new_tokens, True)
        if on:
            rows.append((p, ids, on, reference.greedy(ids, max_new_tokens, False)))
    # Prefer the probes whose answer the adapter changes: elsewhere a server ignoring the adapter agrees anyway.
    rows = [r for r in rows if r[2] != r[3]] or rows
    hf_on, hf_off, srv_on, srv_off, positions = [], [], [], [], []
    for p, ids, on, _ in rows:
        competitors = list(p.competitors) + list(p.stale)
        hf_right = exclusive_match(reference.decode(on), p.gold, competitors)
        srv_right = exclusive_match(client.generate(name, ids, max_new_tokens), p.gold, competitors)
        result.reference_right += hf_right
        result.server_right += srv_right
        result.net_loss += int(hf_right and not srv_right) - int(srv_right and not hf_right)
        full = list(ids) + list(on)
        hf_on += reference.logprobs(full, len(ids), True)
        hf_off += reference.logprobs(full, len(ids), False)
        srv_on += client.prompt_logprobs(name, full, len(ids))
        srv_off += client.prompt_logprobs(None, full, len(ids))
        positions += list(range(len(on)))
    result.probes, result.tokens = len(rows), len(positions)
    if result.net_loss > gates.parity_answer_max_net_loss:
        result.reasons.append(
            f"the server answers {result.net_loss} more probes wrong than HF + PEFT "
            f"(needs <= {gates.parity_answer_max_net_loss})"
        )
    effect_hf = [a - b for a, b in zip(hf_on, hf_off)]
    effect_srv = [a - b for a, b in zip(srv_on, srv_off)]
    corr, s_hf, s_srv = _pearson(effect_hf, effect_srv)
    result.effect_corr = None if corr is None else round(corr, 4)
    if corr is None:
        result.reasons.append(
            "the adapter does not change HF's logprobs on these probes: nothing to verify"
            if s_hf < 1e-6
            else "the server's logprobs do not depend on the adapter (not applied, a wrong module prefix, or "
            "dropped modules)"
        )
    elif corr < gates.parity_effect_corr_min:
        result.reasons.append(f"adapter effect correlation {corr:.3f} < {gates.parity_effect_corr_min}")
    result.k3_adapter = k3_mismatch(hf_on, srv_on, positions)
    result.k3_base = k3_mismatch(hf_off, srv_off, positions)
    if result.k3_adapter is not None and result.k3_base is not None:
        result.k3_ratio = round(result.k3_adapter / max(result.k3_base, gates.parity_k3_floor), 4)
        if result.k3_ratio > gates.parity_k3_ratio_max:
            result.reasons.append(
                f"k3 mismatch with the adapter is {result.k3_ratio:.2f}x the base model's own "
                f"(needs <= {gates.parity_k3_ratio_max})"
            )
    result.passed = not result.reasons
    return result


# ---------------------------------------------------------------------------- publish, rollback, unload


@dataclass
class PublishResult:
    scope: str
    action: str  # publish | rollback | unchanged
    name: str | None = None
    version: int | None = None
    previous: str | None = None
    composite: str = ""
    servers: list = field(default_factory=list)
    canary: list = field(default_factory=list)  # CanaryResult per server
    forced: bool = False
    reasons: list = field(default_factory=list)
    unloaded: list = field(default_factory=list)

    def summary(self) -> str:
        if self.action == "unchanged":
            return f"{self.scope}: {self.name} already serves this chain on {', '.join(self.servers)}"
        verb = "rolled back to" if self.action == "rollback" else "published"
        line = f"{self.scope}: {verb} {self.name} on {len(self.servers)} server(s)"
        if self.previous:
            line += f" (was {self.previous}; it drains, then unloads)"
        if self.forced:
            line += f"; FORCED past: {'; '.join(self.reasons)}"
        return "\n".join([line] + [f"  canary {c.line()}" for c in self.canary])


class Publisher:
    """Publishes a store's scopes to the servers of `serving` (or `servers`). `clients` maps a URL to a client
    (tests, custom transports); `reference` builds the canary's HF + PEFT reference from a composite path."""

    def __init__(
        self,
        learner,
        servers: list[str] | None = None,
        engine: str | None = None,
        transport: str | None = None,
        clients: dict | None = None,
        reference: Callable[[Path], Reference] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.learner, self.store = learner, learner.store
        self.config = learner.config.serving
        self.servers = [u.rstrip("/") for u in (servers if servers is not None else self.config.servers)]
        self.engine = engine or self.config.engine
        self.transport = transport or self.config.transport
        if self.transport == "resolver" and self.engine != "vllm":
            raise PublishError("transport resolver is the vLLM resolver plugin: use it with engine vllm")
        self.base_name = self.config.base_name or self.store.state["base"]
        self._clients = {u.rstrip("/"): c for u, c in (clients or {}).items()}
        self._reference = reference
        self.clock = clock

    def client(self, url: str, engine: str | None = None) -> ServerClient:
        url = url.rstrip("/")
        if url not in self._clients:
            key = os.environ.get(self.config.api_key_env, "") if self.config.api_key_env else ""
            self._clients[url] = make_client(engine or self.engine, url, self.base_name, key, self.config.timeout)
        return self._clients[url]

    def reference(self, composite: Path) -> Reference:
        if self._reference is not None:
            return self._reference(composite)
        return HFReference(
            self.store.state["base"],
            composite,
            self.learner.config.model.device,
            self.learner.config.model.chat_template_kwargs,
        )

    def server_path(self, path: Path) -> str:
        """`path` (under the store) as the servers see it."""
        if not self.config.server_store:
            return str(Path(path).resolve())
        return str(Path(self.config.server_store) / Path(path).resolve().relative_to(self.store.root.resolve()))

    def _canary(self, chain: list, composite: Path, name: str, urls: list[str]) -> list[CanaryResult]:
        rng = random.Random(self.learner.config.eval.seed)
        probes = canary_probes(self.store, chain, self.config.canary_probes, rng)
        reference = self.reference(composite) if probes else None
        system = (self.learner.config.model.system_prompts or [""])[0]
        out = []
        for url in urls:
            if reference is None:
                out.append(run_canary([], self.client(url), None, name, self.learner.config.gates))
                continue
            out.append(
                run_canary(
                    probes,
                    self.client(url),
                    reference,
                    name,
                    self.learner.config.gates,
                    system,
                    self.config.canary_max_new_tokens,
                )
            )
        return out

    def _load(self, entry: dict, path: Path, urls: list[str]) -> list[str]:
        """Load `entry` on `urls` and warm it; on a failure, unload it where it was loaded and raise."""
        loaded = []
        try:
            for url in urls:
                client = self.client(url, entry["engine"])
                client.check_base()
                if entry["transport"] == "api":
                    client.load(entry["name"], self.server_path(path))
                loaded.append(url)
                entry["servers"][url] = "loaded"
                client.generate(entry["name"], "Hello", 1)  # warm-up; the resolver loads the adapter here
        except Exception:
            self._unload_quietly(entry, loaded)
            raise
        return loaded

    def _unload_quietly(self, entry: dict, urls: list[str]) -> None:
        for url in urls:
            try:
                self.client(url, entry["engine"]).unload(entry["name"])
                entry["servers"][url] = "unloaded"
            except PublishError as e:
                entry["servers"][url] = f"unload failed: {e}"

    def _alias_reset(self, scope: str, urls: list[str], engine: str) -> None:
        """Unload the bare scope name where the resolver plugin pinned it, so its next request resolves the
        new live version (under that version's name, so no prefix-cache block is shared)."""
        if engine != "vllm":
            return
        for url in urls:
            try:
                self.client(url, engine).unload(scope)
            except PublishError:
                pass

    def publish(
        self, scope, force: bool = False, canary: bool = True, drain_seconds: int | None = None
    ) -> PublishResult:
        scope = ScopeId.parse(scope)
        if not self.servers:
            raise PublishError("no server to publish to: pass --servers URL ... or set serving.servers")
        drain = self.config.drain_seconds if drain_seconds is None else drain_seconds
        with self.store.lock():
            self.unload()  # versions whose drain time has passed, and forgotten ones
            chain, reasons = publish_check(self.store, scope)
            if reasons and not force:
                raise PublishError(f"{scope} cannot be published: {'; '.join(reasons)} (--force publishes, recorded)")
            if not canary:
                reasons = reasons + ["published without the parity canary (--no-canary)"]
            composite = self.learner.stack(scope)
            cmanifest = json.loads((composite / "manifest.json").read_text())
            published = self.store.published()
            rec = scope_record(published, scope)
            live = rec["live"]
            if live is not None:
                current = rec["versions"][str(live)]
                same = current["composite_id"] == cmanifest["id"] and current["engine"] == self.engine
                if same and all(current["servers"].get(u) == "loaded" for u in self.servers):
                    return PublishResult(str(scope), "unchanged", current["name"], live, servers=self.servers)
            n = rec["next"]
            rec["next"] = n + 1
            entry = {
                "name": served_name(scope, n),
                "composite": str(composite.relative_to(self.store.root)),
                "composite_id": cmanifest["id"],
                "chain": [list(c) for c in cmanifest["chain"]],
                "rank": cmanifest["rank"],
                "engine": self.engine,
                "transport": self.transport,
                "created": now(),
                "state": "loading",
                "servers": {},
                "forced": bool(reasons),
                "forced_reasons": reasons,
                "canary": None,
            }
            result = PublishResult(
                str(scope), "publish", entry["name"], n, composite=entry["composite"], servers=list(self.servers)
            )
            result.forced, result.reasons = bool(reasons), reasons
            # Recorded before loading: the resolver plugin finds the version by its name in published.json.
            rec["versions"][str(n)] = entry
            self.store.write_published(published)
            try:
                self._load(entry, engine_adapter(composite, self.engine), self.servers)
                if canary:
                    result.canary = self._canary(entry["chain"], composite, entry["name"], self.servers)
                    entry["canary"] = {c.server: c.to_dict() for c in result.canary}
                    failed = [c for c in result.canary if not c.passed]
                    if failed:
                        raise CanaryError(
                            f"{entry['name']} failed the parity canary: "
                            + " | ".join(f"{c.server}: {'; '.join(c.reasons)}" for c in failed)
                            + "; it was unloaded and the live version is unchanged"
                        )
            except Exception as e:
                self._unload_quietly(entry, [u for u, s in entry["servers"].items() if s == "loaded"])
                entry["state"], entry["error"] = "failed", str(e)[:1000]
                rec["versions"][str(n)] = entry
                rec["history"].append({"at": now(), "action": "publish_failed", "version": n, "reason": str(e)[:500]})
                self.store.write_published(published)
                raise
            entry["state"], entry["published_at"] = "live", now()
            if live is not None:
                old = rec["versions"][str(live)]
                old["state"], old["unload_after"] = "draining", self.clock() + drain
                result.previous = old["name"]
            rec["live"] = n
            rec["versions"][str(n)] = entry
            rec["history"].append(
                {"at": now(), "action": "publish", "version": n, "from": live, "forced": entry["forced"]}
            )
            self.store.write_published(published)
            self._alias_reset(str(scope), self.servers, self.engine)
            if drain == 0:
                result.unloaded = self.unload(scope, everything_drained=True)
        return result

    def rollback(
        self, scope, to: int | None = None, canary: bool = False, drain_seconds: int | None = None
    ) -> PublishResult:
        """Serve an earlier published version of `scope` again (default: the newest one before the live one)."""
        scope = ScopeId.parse(scope)
        drain = self.config.drain_seconds if drain_seconds is None else drain_seconds
        with self.store.lock():
            published = self.store.published()
            rec = published["scopes"].get(str(scope))
            if not rec or rec.get("live") is None:
                raise PublishError(f"{scope} has no published live version to roll back from")
            live = rec["live"]
            usable = sorted(int(n) for n, v in rec["versions"].items() if v["state"] in LIVE_STATES and int(n) != live)
            if to is None:
                earlier = [n for n in usable if n < live]
                if not earlier:
                    raise PublishError(f"{scope}: no earlier published version to roll back to")
                to = earlier[-1]
            entry = rec["versions"].get(str(to))
            if entry is None or int(to) not in usable:
                state = entry["state"] if entry else "never published"
                raise PublishError(f"{scope}: cannot roll back to v{to} ({state})")
            for s, version in entry["chain"]:
                if _gone(self.store, s, version):
                    raise PublishError(f"{scope}: v{to} holds {s} {version}, which was forgotten (pruned)")
            composite = self.store.root / entry["composite"]
            if not (composite / "manifest.json").exists():
                composite = self.learner.stack(scope, chain=entry["chain"])
            urls = list(entry["servers"]) or self.servers
            todo = [u for u in urls if entry["servers"].get(u) != "loaded"]
            if todo:
                self._load(entry, engine_adapter(composite, entry["engine"]), todo)
            result = PublishResult(
                str(scope), "rollback", entry["name"], to, composite=entry["composite"], servers=urls
            )
            if canary:
                result.canary = self._canary(entry["chain"], composite, entry["name"], urls)
                if any(not c.passed for c in result.canary):
                    self._unload_quietly(entry, todo)
                    self.store.write_published(published)
                    raise CanaryError(f"{entry['name']} failed the parity canary; the live version is unchanged")
            old = rec["versions"][str(live)]
            old["state"], old["unload_after"] = "draining", self.clock() + drain
            entry["state"] = "live"
            entry.pop("unload_after", None)
            rec["live"] = to
            rec["history"].append({"at": now(), "action": "rollback", "version": to, "from": live})
            result.previous = old["name"]
            own = dict((s, v) for s, v in entry["chain"]).get(str(scope))
            revert_live(self.store, scope, own)
            self.store.write_published(published)
            self._alias_reset(str(scope), urls, entry["engine"])
            if drain == 0:
                result.unloaded = self.unload(scope, everything_drained=True)
        return result

    def unload(self, scope=None, everything_drained: bool = False) -> list[str]:
        """Unload the versions whose drain time has passed (every draining one with everything_drained), and every
        version that holds a forgotten (pruned) adapter, from the servers they were loaded on. Unload errors are
        recorded and retried by the next call. Returns lines describing what was unloaded."""
        with self.store.lock():
            published = self.store.published()
            lines, t = [], self.clock()
            for s, rec in published["scopes"].items():
                if scope is not None and s != str(ScopeId.parse(scope)):
                    continue
                for n, entry in rec["versions"].items():
                    if entry["state"] == "loading":
                        # A publish that died between loading and the flip (the store lock is held here, so it
                        # is not running): it never went live.
                        self._unload_quietly(entry, [u for u, x in entry["servers"].items() if x == "loaded"])
                        entry["state"], entry["error"] = "failed", "interrupted before the flip"
                        lines.append(f"{entry['name']}: an interrupted publish, marked failed")
                        continue
                    pruned = any(_gone(self.store, cs, cv) for cs, cv in entry["chain"])
                    due = entry["state"] == "draining" and (everything_drained or t >= entry.get("unload_after", 0))
                    if not (pruned or due):
                        continue
                    for url, state in list(entry["servers"].items()):
                        if state == "loaded" or state.startswith("unload failed"):
                            try:
                                self.client(url, entry["engine"]).unload(entry["name"])
                                entry["servers"][url] = "unloaded"
                                lines.append(f"unloaded {entry['name']} from {url}")
                            except PublishError as e:
                                entry["servers"][url] = f"unload failed: {e}"
                                lines.append(f"could not unload {entry['name']} from {url}: {e}")
                    if pruned:
                        entry["state"] = "pruned"
                        if rec["live"] == int(n):
                            rec["live"] = None
                            rec["history"].append({"at": now(), "action": "forgotten", "version": int(n)})
                            self._alias_reset(s, list(entry["servers"]), entry["engine"])
                    elif all(v == "unloaded" for v in entry["servers"].values()):
                        entry["state"] = "retired"
            self.store.write_published(published)
        return lines


def _gone(store: Store, scope: str, version: str) -> bool:
    """The version can no longer be served: forgotten (pruned), or its manifest is missing."""
    path = store.scope(scope).version_dir(version) / "manifest.json"
    return not path.exists() or bool(json.loads(path.read_text()).get("pruned"))


def revert_live(store: Store, scope: ScopeId, version: str | None) -> None:
    """Point the scope's LIVE at `version` after a rollback: what later versions consolidated is unconsolidated
    again (served in the prompt, trained by the next session), and their replay prompts leave the pool."""
    ss = store.scope(scope)
    if ss.live == version:
        return
    later = [v for v in ss.versions() if version is None or v > version]
    sessions = {ss.manifest(v).get("session") for v in later}
    later = set(later)
    notes, facts = ss.notes(), ss.facts()
    for item in notes + facts:
        if item.consolidated_in in later:
            item.consolidated_in = None
    ss.write_notes(notes)
    ss.write_facts(facts)
    ss.write_pool([r for r in ss.pool() if r.get("session") not in sessions])
    ss.set_live(version)


def published_summary(store: Store) -> list[dict]:
    """Per published scope: the live name, its servers and the versions still loaded somewhere."""
    out = []
    for s, rec in sorted(store.published()["scopes"].items()):
        live = rec["versions"].get(str(rec["live"])) if rec.get("live") is not None else None
        out.append(
            {
                "scope": s,
                "live": live["name"] if live else None,
                "servers": sorted(live["servers"]) if live else [],
                "forced": bool(live and live.get("forced")),
                "forced_reasons": list(live.get("forced_reasons") or []) if live else [],
                "loaded": sorted(
                    v["name"] for v in rec["versions"].values() if any(x == "loaded" for x in v["servers"].values())
                ),
            }
        )
    return out


__all__ = [
    "CanaryError",
    "HFReference",
    "PublishError",
    "Publisher",
    "RuntimeLoRAError",
    "parse_name",
    "publish_check",
    "served_name",
    "served_version",
    "wake_notes",
]
