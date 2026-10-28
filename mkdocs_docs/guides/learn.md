# Continual Learning (experimental)

!!! warning "Experimental"
    `pgs learn` lives on a feature branch. Its interfaces and defaults may change, and every recipe default
    is a hypothesis until a benchmark settles it. Each trained version is evaluated and gated before it can
    go live, but the gate thresholds are **uncalibrated** defaults until a calibration on null sessions sets
    them. Serving (publishing to vLLM or SGLang, the resolver plugin, the proxy) is tested on CPU against
    fake servers; the GPU integration tests have not run yet.

`palingenesis.learn` keeps what a deployed model must know per **scope** (an organization, a team, a person)
and consolidates it into one small LoRA adapter per scope.

```bash
pip install 'palingenesis[learn]'
pgs learn init stores/acme --base Qwen/Qwen3.5-4B --org acme
pgs learn add stores/acme --scope org:acme --documents "docs/**/*.md" --approve
pgs learn plan stores/acme                      # what the next run trains, and its estimated cost
pgs learn run stores/acme --scope org:acme --live   # train, evaluate, go live if the gates pass
pgs learn publish stores/acme --scope org:acme --servers http://10.0.0.5:8000   # org:acme@v1, canary first
pgs learn status stores/acme
```

## Why this matters

A small model deployed for one company, team or person needs to know things no base model knows: this week's
product changes, the team's conventions, the user's preferences, and the corrections people made to its
answers. Today the options are:

- **Retrieval (RAG)**: good for large, fast-changing corpora and citations, but every request pays for
  retrieval and context. Small models also use long, noisy contexts less reliably than large ones.
- **Long system prompts**: simple, but the cost grows with every fact, and a small model's context budget
  fills fast.
- **Periodic full fine-tuning**: forgets earlier updates, can damage general ability, and gives one model per
  tenant, which cannot be served at scale.

`pgs learn` adds a fourth option that works alongside retrieval. Knowledge that is used often and changes
slowly is consolidated into a small LoRA adapter per scope. New items live in the prompt until the next
training session; the adapters of a scope chain stack into one, so one base model on one inference server can
serve many scopes. Deleting an item deletes it: its text is purged from the store and the adapter versions
that learned it are pruned.

What it is not: a replacement for retrieval over large corpora, a way to learn skills from traffic, or a
guarantee of zero forgetting.

## Concepts

### Scopes

```
base model (frozen, shared)
└── org:acme                       adapter rank 32
    ├── team:acme/support          adapter rank 16
    │   ├── user:acme/support/ana  adapter rank 16
    │   └── user:acme/support/luca
    └── team:acme/sales
```

- A store holds one org (one tenant) and everything below it. Stores never share files across orgs.
- Each scope has its own sources, notes, facts, inbox, replay pool and adapter versions.
- A scope's adapter learns **only that scope's notes**. It is trained on top of its ancestors' live adapters,
  held frozen, so it stores only the difference its own notes make.
- The adapters of a chain (org + team + user) stack exactly into one LoRA of the summed rank (32 + 16 + 16 =
  64), padded with zero rank up to a rank vLLM accepts.
- A team or org note can be **locked** (policy): a lower scope cannot contradict it.

### Notes and facts

- A **note** is what is stored, rendered into prompts and forgotten: a document chunk, a fact, a correction, a
  remembered item, or an instruction (behaviour or policy, team and org only).
- An **atomic fact** is what is trained: one subject, attribute and value of a note. Questions are generated
  per fact.
- A document source has a stable key (its path under the source root). Refreshing it diffs its chunks: an
  unchanged chunk keeps its note, a changed chunk supersedes it (its changed facts record the old value as
  stale), a moved chunk keeps its id, a removed chunk is retired.
- New facts are checked against the active facts of the scope and its ancestors. A likely conflict waits in
  the inbox (supersede, keep both, or reject); a conflict with a locked ancestor note is rejected.

### Learning signals

Only deliberate acts create notes:

| signal | from | scope | consent |
|---|---|---|---|
| explicit command | the newest user message starting with `remember`, `ricorda`, `recuerda`, `souviens-toi`, `merk dir` (configurable) | user | the command itself |
| correction | an explicit call with the prompt and the corrected answer | user | the call |
| note or document | `pgs learn add` | user, or team / org by an admin | the scope's inbox, or `--approve` |
| promotion | `pgs learn promote` | team / org | the admin queue |

Never a signal: assistant, system and tool messages, tool results inside a user message, earlier turns, text in
code fences, blockquotes or quotes, and anything after the command's first line (pasted content). A remembered
note that reads like an instruction ("always", "ignore", "you must") waits for confirmation.

### PII filter

Every note passes a PII filter before anything is written, the inbox included.

| category | examples | user scope | team / org scope |
|---|---|---|---|
| secret | API keys, private keys, passwords, card numbers (Luhn), IBANs (checksum) | rejected | rejected |
| personal | emails, phone numbers, Italian codice fiscale and partita IVA, US social security numbers, street addresses | kept (default) | redacted (default) |

`pii.user_scope_personal` and `pii.shared_scope_personal` take `allow`, `redact` or `block` (held for review).
The regex detectors are a baseline: emails; international, Italian mobile and landline numbers (spaced
forms included; amounts, dates and ISBNs are not taken for phones); the codice fiscale and partita IVA
(with their check characters); US SSNs; a basic street-address pattern (`via Roma 12`, `221 Baker Street`).
Pass your own detectors (any object with `find(text)`, such as a wrapper around Presidio) to `PIIFilter`.

## Quickstart

```bash
# A store for one org, on the base model you serve
pgs learn init stores/acme --base Qwen/Qwen3.5-4B --org acme

# Org documents, approved by the admin running the command
pgs learn add stores/acme --scope org:acme --documents "**/*.md" --root acme-docs --approve

# A user's notes wait in their inbox until approved
pgs learn add stores/acme --scope user:acme/support/ana --note "My project is Orion."
pgs learn inbox stores/acme --scope user:acme/support/ana
pgs learn approve stores/acme --scope user:acme/support/ana i_3f2a...

# What the next run would train and evaluate, and its cost (measured on this store once a session ran)
pgs learn plan stores/acme

# Train the scope's adapter on its new facts and evaluate it against the live version;
# --live makes it live only if every gate passes
pgs learn run stores/acme --scope org:acme --live
pgs learn run stores/acme --scope user:acme/support/ana --live --recipe.passes 2

# Evaluate again (exit 1 if a gate fails); --live --force publishes past a failed gate, recorded
pgs learn eval stores/acme --scope org:acme --version v0003
pgs learn status stores/acme                  # exit 1 on a pending rebuild or a forced live version

# Next week: refreshed documents supersede the changed chunks only
pgs learn add stores/acme --scope org:acme --documents "**/*.md" --root acme-docs --approve
pgs learn run stores/acme --scope org:acme --live

# Forget a note everywhere (published versions that held it are unloaded from the servers)
pgs learn forget stores/acme --scope user:acme/support/ana n_8c1d...

# Serve: publish the stacked chain under a new versioned name, roll back, run everything nightly
pgs learn publish stores/acme --scope user:acme/support/ana --servers http://10.0.0.5:8000
pgs learn rollback stores/acme --scope user:acme/support/ana
pgs learn sleep stores/acme --publish --budget-minutes 240        # from cron
pgs learn proxy stores/acme --keys keys.yaml --upstream http://10.0.0.5:8000
```

The same from Python:

```python
from palingenesis.learn import Learner

learner = Learner.init("stores/acme", base="Qwen/Qwen3.5-4B", org="acme")
learner.scope("org:acme").add_documents("**/*.md", root="acme-docs", approve=True)
learner.intake("user:acme/support/ana", [{"role": "user", "content": "ricorda che preferisco risposte brevi"}])
result = learner.run("org:acme")               # trains, then evaluates against the live version
print(result.evaluation.summary())
learner.set_live("org:acme", result.version)   # StoreError if a gate failed (force=True overrides)
composite = learner.stack("user:acme/support/ana")   # org + team + user as one PEFT adapter
learner.publish("org:acme", servers=["http://10.0.0.5:8000"])   # org:acme@v1, after the parity canary
messages = learner.render_notes("user:acme/support/ana", messages)   # notes not consolidated yet
report = learner.forget("user:acme/support/ana", ["n_8c1d..."])
assert report.ok
```

Between sessions, `render_notes` puts the chain's unconsolidated notes into the system turn as quoted
reference data, in the same format the training teacher sees. Once something is published, "unconsolidated"
means not held by the published composite the scope is served with (the proxy uses the same rule).

## How a session trains

`pgs learn run` trains one scope. P, the "parent stack", is the base plus the ancestors' live adapters; the
student is P plus the scope's adapter, warm-started from its live version.

| row | student prompt | teacher (default) | notes in the teacher's system turn |
|---|---|---|---|
| new | a question per new fact | P + notes | the fact's note and its BM25 neighbours |
| replay | a stored prompt | P + the scope's current notes | the notes its facts resolve to now |
| near-miss | the same question about a real entity | P + relevant notes | notes scoring above `notes.relevance_threshold` |
| anchor | a general prompt | P + relevant notes | plus every instruction note of the scope |

- The default `notes.relevance_threshold` of 0.0 admits any note sharing a word with the prompt, so most
  near-miss and many anchor rows carry notes; the threshold is to be calibrated.
- Questions per fact: direct, paraphrase, reverse direction, and one combined with a sibling fact. Probes are
  written with a different template and kept out of training (token-set Jaccard below 0.5 against every
  train question).
- The teacher always sees the **current** notes, so a replayed prompt about a superseded fact is answered from
  the new note, and no answer is stored as a label.
- The loss is GKD (Agarwal et al., arXiv 2306.13649): `recipe.onpolicy_fraction` of the rows continue the
  student's own sample, the rest a teacher sample, with a full-vocabulary divergence on every token
  (`recipe.divergence`: `fkl` default, `rkl`, `jsd`). `recipe.mode: sft` trains cross-entropy on teacher
  samples instead. Teacher samples that mention the notes ("according to the notes") are dropped.
- A preflight asks the teacher the probes with the notes in context and aborts when it fails too many: the
  model's prior then wins over the note, and training would not help. Rewrite the note.
- The adapter is a LoRA on the MLP projections only (`gate_proj`, `up_proj`, `down_proj` of the decoder
  layers, matched by an anchored regex so vision towers and MTP heads never match). Factual associations are
  largely stored in mid-layer MLPs (ROME, arXiv 2202.05262; MEMIT, arXiv 2210.07229; Geva et al., arXiv
  2012.14913). Whether multi-hop or document-level facts need attention targets is an open question.

Each session writes `sessions/sNNNN/` (questions, probes, near-miss and abstain prompts, config, losses,
and the evaluation's report) and `adapters/vNNNN/` (a PEFT adapter and a manifest listing the notes active
during training, the parent versions it was trained on, and its evaluation). A failed session leaves no
files behind.

!!! note "What the CPU tests show"
    The test suite runs the whole loop on tiny random models. It checks mechanics (rows, teachers, frozen
    ancestors, stacking, crash safety), not that facts are learned.

## Reports and gates

Every metric compares two chains on the same probes, greedy, with the deployment system prompt: the
**candidate** (the ancestors' live adapters plus the new version) and the **live** chain (the ancestors'
live adapters plus the scope's live version, or none yet). The in-context ceiling is the parent stack with
the notes in the prompt and no adapter of the scope.

| metric | probes | measure |
|---|---|---|
| acquisition | held-out probes of the session's new facts | answered closed-book, among the probes the in-context ceiling answers |
| update | probes of facts that superseded an older value | the new value, same measure |
| stale | probes of every superseded fact of the scope | answers stating an old value |
| retention | probes of the scope's earlier, consolidated facts | paired counts vs live: lost, gained, net loss |
| ancestor retention | probes of the ancestors' facts, asked of the whole chain | paired counts vs live |
| general QA | a short multilingual set (`eval.general_qa_path` to replace it) | paired counts vs live |
| leak | near-miss probes (the same question about a real entity) and general QA | answers that changed vs live, are wrong, and name an entity of the scope that the live answer did not |
| confabulation | questions the notes do not answer: an unstated attribute, an invented sibling (`Orion Ultra`) | answered instead of declined, paired vs live |
| meta-reference | every closed-book answer, and the reasoning of a thinking model | mentions of "my notes", "the context", "secondo le mie note" |
| drift | fixed general prompts | KL(target ‖ model) per token, target = the anchor teacher (the parent stack plus relevant and instruction notes); the increment vs live and the absolute value |

- Answers are scored with an **exclusive matcher**: the answer states the expected value and none of its
  competitors (the fact's stale values, the values other entities have for the same attribute, and for
  general QA the scope's own entities). "25 or 30 euros" is wrong.
- The decline detector covers English, Italian, Spanish, French and German; pass `decline=` to
  `Learner.evaluate` to use a judge instead.
- Rates are shown with Wilson 95% intervals; the gates use the counts. Paired counts come with an exact
  McNemar p-value, for reading only.
- The live chain's answers are cached in the session's `answers.jsonl`, so they are reused from the
  evaluation that made the live version a candidate, and only new probes are asked of both.

Each evaluation writes `report.json` (counts, rates, gates; no note text), `report.md` (a summary with the
failed probes, their answers and the note text, so a note can be fixed) and `diagnostics.jsonl` (every
failed probe). The manifest of the version records the outcome.

| gate | default | passes when |
|---|---|---|
| `acquisition_min` | 0.80 | acquisition (relative to the in-context ceiling) is at least this |
| `update_min` | 0.80 | update is at least this |
| `stale_max` | 0.05 | stale is at most this |
| `retention_max_net_loss` | 3 | lost minus gained earlier facts vs live |
| `ancestor_retention_max_net_loss` | 3 | same, on the ancestors' facts |
| `general_qa_max_net_loss` | 5 | same, on general QA |
| `leak_max_net` | 2 | leaking answers |
| `confab_max_net` | 2 | newly confabulated minus newly declined |
| `meta_reference_max` | 0.02 | share of answers mentioning notes |
| `drift_increment_max` | 0.03 | KL increment vs live, nats per token |
| `drift_cap` | 0.15 | KL to the target, nats per token |

A gate with nothing to measure (no superseded fact, no earlier fact) is skipped. A failed gate blocks
`set_live` / `--live`, and the notes stay in the prompt (`render_notes`); `--force` publishes anyway and
the manifest records it with the reasons, which `status` reports as an issue. A version evaluated against
a live chain that has changed since must be evaluated again.

!!! warning "Uncalibrated thresholds"
    The defaults above are not calibrated (`gates.calibrated: false`, shown in every report). They will be
    set from null sessions (the same session rerun with another seed, notes already known, replay and
    anchors only) to a per-gate false-fail rate, with the family-wise rate published. Until then, read a
    failed gate together with its counts, and expect some false fails, especially on small scopes.

`pgs learn plan` counts what the next run of each scope would train (new facts, questions, near-miss,
anchor and replay rows, steps) and evaluate, checks the stacked rank against vLLM's accepted ranks, and
estimates the time from the last session measured on the same store. `pgs learn status` prints the scope
tree with live versions, gate states, unconsolidated facts, inbox and admin-queue counts, pending rebuilds
and the retention trend per session (`--json` for scripts; exit 1 when something needs a person).

## Forget

`pgs learn forget STORE --scope S NOTE...`:

1. **Purge**: the note's text and everything derived from it (its facts, their questions, probes and replay
   prompts, inbox and admin-queue entries, earlier versions in its supersede chain, stale values) are removed
   from every file of the scope. Files are rewritten, not appended; `tombstones.jsonl` keeps only the id and a
   hash.
2. **Prune**: every adapter version trained while the note was active, or warm-started from such a version,
   loses its weights, and every stacked composite built from such a version is deleted. The scope, and descendants trained on a pruned parent, are
   marked for rebuild: run `pgs learn run` again.
3. **Verify**: every file of the store is searched for the note's text and its facts' sentences (pieces of 12
   characters or more), and every text file for its facts' subjects and values of any length as whole words
   ("Kestrel", "42 kg"). Trivial pieces (under 3 characters, common words, numbers of fewer than 4 digits)
   and pieces that a kept note, fact or inbox item still states are not searched. The command exits with 1
   and lists the files if any still hold one.

Session rows derived from the note (questions, probes, near-miss and abstain prompts, cached evaluation
answers, diagnostics), in the scope and in its descendants, are purged too, and `report.md` is rendered again.

## Serving

A scope is served as **one** LoRA: the live adapters of its chain (org + team + user) stacked exactly into
one PEFT adapter of the summed rank, padded with zero rank to a rank vLLM accepts (`--max-lora-rank`: 1, 8,
16, 32, 64, 128, 256, 320 or 512). One base model on one server serves every scope of every tenant.

### Versioned names

Every publish is a **new name**, `<scope>@v<n>` (`user:acme/support/ana@v3`), where n counts the scope's
publishes. Nothing is reloaded in place:

- vLLM keys its prefix cache by adapter name, so KV blocks (and the recurrent-state checkpoints of hybrid
  models) computed under one version are never reused under another;
- requests already running finish on the version they started with;
- the old version stays loaded for `serving.drain_seconds` (default 60), then is unloaded.

Clients do not need these names: the proxy routes each API key to its scope's live name, and
`pgs learn status --json` shows it (`published.live`) for clients that call the server directly.

### Publish

```bash
pgs learn publish stores/acme --scope user:acme/support/ana \
  --servers http://10.0.0.5:8000 http://10.0.0.6:8000      # every replica
```

1. **Gates**: publishing is refused when a version of the chain went live forced or ungated, failed its
   evaluation, or was evaluated on other parent versions than the parents' live ones (evaluate it again).
   `--force` publishes anyway; the record keeps the reasons and `status` reports them.
2. **Load**: the composite is built (or reused) under `composites/` and loaded on every server under the new
   name: `POST /v1/load_lora_adapter` on vLLM, `POST /load_lora_adapter` on SGLang, or nothing with
   `--transport resolver` (the vLLM plugin below loads it on the first request). Each server must serve the
   store's base model (`serving.base_name` if it names it differently). A one-token request warms it.
3. **Canary**: the parity canary (below) runs on every server. If any server fails, or any load fails, the
   new name is unloaded from every server and the live version stays as it was.
4. **Flip**: `published.json` records the new live version; the proxy routes to it on the next request.

`pgs learn rollback STORE --scope S [--to N]` serves an earlier version again (loaded again if it was
unloaded, `--canary` to check it first). The scope's `LIVE` follows the rollback, and what the rolled-back
versions consolidated counts as unconsolidated again: those notes return to the served prompt and the next
session trains them. A version that held a forgotten note cannot be rolled back to.

`pgs learn unload STORE` unloads the versions whose drain time has passed; `publish`, `rollback` and `sleep`
do it too, so no daemon is needed. `forget` unloads every published version that held a forgotten note.

### Parity canary

The canary checks that the server applies the adapter as HF + PEFT does, on the probes where it matters.

- **Reference**: transformers + PEFT loaded with the same composite (not another vLLM run, which would share
  the loader and miss a key-name bug).
- **Probes**: probes of the facts the chain consolidated, at most `serving.canary_probes`, preferring those
  whose greedy answer the adapter changes (elsewhere a server ignoring the adapter would agree anyway).
- **Checks**, per server (thresholds in `gates:`):
    - answers: the probes HF answers right and the server wrong, minus the reverse, at most
      `parity_answer_max_net_loss` (1);
    - adapter effect: per token of HF's answers, logprob with the adapter minus without, on both sides; the
      correlation must be at least `parity_effect_corr_min` (0.9);
    - numerics: the k3 mismatch (`rho - 1 - log rho`, `rho = exp(lp_hf - lp_server)`, the RL trainer's
      sampler-mismatch metric) with the adapter, divided by the base model's own k3 on the same tokens, at most
      `parity_k3_ratio_max` (1.5). Hybrid models already differ between HF and the server (fp32 head, recurrent
      state dtype, norm rounding); the ratio blames the export only for what the adapter adds.
- A server that ignores the adapter, a wrong module prefix or a dropped module fails it. `--no-canary`
  publishes without it, recorded as a forced publish.
- The reference runs where `publish` runs: it loads the base model with transformers on `model.device`
  (a GPU for anything but small models).

### vLLM

```bash
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
VLLM_PLUGINS=palingenesis_learn PGS_LEARN_STORE=/srv/stores \
  vllm serve Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 64 --max-loras 8 --max-cpu-loras 64 \
  --api-key "$PGS_LEARN_SERVER_KEY" --host 10.0.0.5 --port 8000
```

- `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` enables the load / unload routes and the resolver plugins. vLLM
  warns that this is for trusted, isolated environments: anyone who can reach the server can load an adapter
  from any path the server can read. Keep the server on a private network, behind the proxy, with
  `--api-key` (the publisher sends the key from `serving.api_key_env`, default `PGS_LEARN_SERVER_KEY`).
- Without the variable, `publish` fails with a clear error naming it.
- `--max-lora-rank` is the stacked rank (`pgs learn plan` prints it per scope); `--max-loras` is about the
  number of scopes active in one batch, `--max-cpu-loras` the scopes active within a few minutes.
- Run one API server process (`--api-server-count 1`, the default): vLLM tracks runtime-loaded names per
  front-end process.
- The adapter is trained in fp32 and served in the server's LoRA dtype (bf16 or fp16); the canary measures the
  effect.
- `serving.server_store` maps the store's path to the path the servers see (a shared volume).

**Resolver plugin.** `pip install 'palingenesis[learn]'` in the vLLM environment registers the
`palingenesis_learn` plugin (`vllm.general_plugins` entry point). With `VLLM_PLUGINS=palingenesis_learn` and
`PGS_LEARN_STORE` (a store, a directory of stores, or several separated by `:`), a request for
`<scope>@v<n>` loads that published composite from the store on demand, after a restart or on a replica
that never received a load call. It resolves only published (not failed) versions whose adapters were not
forgotten, for the base the server runs (`PGS_LEARN_CHECK_BASE=0` skips that check), and stacks a missing
composite again. A request for the bare `<scope>` gets the live version; vLLM then keeps that name pinned to
it until it is unloaded, which `publish` does on the servers it publishes to (the version is loaded under its
own versioned name, so no cached block crosses versions). Use `--transport resolver` to publish without
calling the load route.

### SGLang

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --enable-lora --max-lora-rank 64 \
  --lora-target-modules gate_up_proj down_proj --max-loras-per-batch 4 --api-key "$PGS_LEARN_SERVER_KEY"
pgs learn publish stores/acme --scope org:acme --engine sglang --servers http://10.0.0.7:30000
```

- SGLang's `/load_lora_adapter` and `/unload_lora_adapter` share the inference port and have no runtime
  guard: run it with `--api-key` on a private network.
- The adapter for SGLang is written with SGLang's module names (`model.language_model.` becomes `model.` on
  multimodal checkpoints, as SGLang renames the base weights); the canary verifies it.
- A switch holds two versions of a scope for a moment: size `--max-loras-per-batch` with that in mind.
- The proxy selects the adapter with SGLang's `lora_path` request field.

### What is published where

`published.json` (store root) records, per scope: the live version, every version with its name, composite,
chain, rank, engine, servers and their state (`loaded`, `unloaded`, `unload failed: ...`), the canary result
per server, whether it was forced and why, and a history of publishes, failures, rollbacks and forgotten
versions. It holds names and ids only, no note text.

## The proxy

`pgs learn proxy` is an optional OpenAI-compatible front for one store (standard library only):

```bash
pgs learn key stores/acme --scope user:acme/support/ana --keys keys.yaml   # prints a key once
pgs learn proxy stores/acme --keys keys.yaml --upstream http://10.0.0.5:8000 --port 8100
curl http://proxy:8100/v1/chat/completions -H "Authorization: Bearer $ANA_KEY" \
  -d '{"model": "any", "messages": [{"role": "user", "content": "ricorda che il mio progetto è Orion"}]}'
```

Per request:

1. **Auth**: the key's sha256 maps to a scope, a role and consent flags in `keys.yaml`. The client's key is
   never forwarded; the upstream gets the server key.
2. **Signals**: on a chat request from a user key, an explicit command in the newest user message is stored
   in that key's scope through the PII filter (the rules of [Learning signals](#learning-signals)), at most
   `signals.max_notes_per_hour` per key. The response says what happened: `x-learn-note: stored | held |
   rejected | deferred | rate_limited` and `x-learn-note-id`. While a job holds the store (a session is
   training), the signal is PII-screened and spooled, and the next job stores it.
3. **Notes**: the chain's notes that the served composite does not hold yet are rendered first in the system
   turn, ancestors' first, oldest first: the prefix stays the same between sessions, so the server's prefix
   cache keeps hitting. At most `notes.wake_max_tokens` (about 4 characters per token); `x-learn-notes` and
   `x-learn-notes-dropped` report the counts. Notes past the budget mean the scope should sleep.
4. **Routing**: `model` becomes the scope's live name (or the nearest published ancestor's, or the base model
   when nothing is published). Adapter fields a client sends (`lora_path`, ...) are dropped: a key reaches its
   own scope only. `x-learn-version` names what answered.
5. **Forwarding**: streaming (SSE) and tool calls pass through unchanged.

| route | who | what |
|---|---|---|
| `POST /v1/chat/completions` | any key | routed, notes injected, commands captured (user keys) |
| `POST /v1/completions` | any key | routed only |
| `GET /v1/models` | any key | the key's scope |
| `POST /v1/learn/corrections` `{prompt, correction}` | user keys | a correction note in the key's scope |
| `GET /v1/learn/notes` | any key | the scope's notes and pending items |
| `POST /v1/learn/forget` `{ids}` | user keys | forget the key's own notes |
| `POST /v1/learn/notes` `{notes, kind}` | admin keys | into the scope's inbox (never approved by the proxy) |
| `GET /health` | anyone | liveness |

Everything else, the servers' LoRA load and unload routes included, answers 404. The proxy logs nothing by
default (no prompt, no response); `--access-log` prints method, path, status, scope and latency only. Its
added latency target is p50 below 2 ms.

```yaml
# keys.yaml
keys:
  - sha256: 9f86d081884c7d65...      # of the key; `pgs learn key` appends entries
    scope: user:acme/support/ana
    role: user                       # user | admin
    name: ana                        # the origin recorded on captured notes
    remember: true                   # consent to command capture (default true for user keys)
    corrections: true
```

Without the proxy, `Learner.render_notes(scope, messages)` adds the same notes block client-side, and
`status --json` gives the live name.

## Nightly sleep

```bash
# crontab: every night at 02:00, at most 4 GPU hours
0 2 * * *  pgs learn sleep /srv/stores/acme --publish --budget-minutes 240 >> /var/log/pgs-learn.log 2>&1
```

`pgs learn sleep` is one pass, not a daemon:

1. stores spooled signals; with `--publish`, unloads drained and forgotten versions;
2. orders the scopes with pending work (unconsolidated facts, or a pending rebuild) parents first, then by
   pending facts (`--scopes` limits it);
3. per scope: estimates the time from the store's measured sessions (`plan`), runs and evaluates a session,
   sets it live if every gate passes, and publishes it with the canary (`--publish`). A scope that does not fit
   the remaining budget (`--budget-minutes`, `--scope-budget-minutes`) waits, its notes still in the prompt;
4. with `sleep.children_follow: gated`, each published descendant of a scope that went live is evaluated on
   the new parent and republished if its gates pass; otherwise it keeps serving its previous composite and is
   reported.

A failed gate never goes live. The exit code is 1 when something needs a person (a failed gate or canary, an
error), so cron can mail it.

## Security

- **Runtime LoRA loading is a trust boundary.** With `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, anyone who reaches
  vLLM can load an adapter from a path; SGLang's routes have no switch at all. Servers belong on a private
  network with `--api-key`, reachable by users only through the proxy, which never forwards those routes.
- **Poisoning.** Only deliberate signals of an authenticated key reach its own user scope. Shared scopes
  change only through admins (`pgs learn add`, the admin queue, admin keys into the inbox). A user can still
  teach their own scope wrong things; that is by design, and `forget` undoes it.
- **Prompt injection into memory.** Text the user pasted, quoted or fenced, tool outputs, retrieved documents,
  assistant and system turns and earlier turns are never captured; only the first line of a command is kept.
  A note that reads like an instruction ("always", "ignore", "you must") waits for confirmation. Notes are
  rendered as quoted reference data, never as instructions.
- **PII.** Every note passes the PII filter before it is written, the spool included: secrets are rejected,
  personal identifiers kept, redacted or held per scope. The proxy stores no prompt or response.
- **Keys.** `keys.yaml` holds key hashes only (`pgs learn key` writes it with mode 600). Rotate a key by
  removing its entry and restarting the proxy.
- **Forget** removes a note from every file of the store, prunes the adapter versions and composites that held
  it, and unloads their published names from the servers.

## Tutorial: teach your model your docs, every week

A product team keeps its documentation in `acme-docs/` and wants its assistant to know each week's changes
without retraining a model.

**Once.**

```bash
pip install 'palingenesis[learn]'
pgs learn init stores/acme --base Qwen/Qwen3.5-4B --org acme
pgs learn add stores/acme --scope org:acme --documents "**/*.md" --root acme-docs --approve
pgs learn plan stores/acme                 # facts, rows, steps, and the stacked rank to serve
pgs learn run stores/acme --scope org:acme --live
```

Read `scopes/org/acme/sessions/s0001/report.md`: acquisition relative to the in-context ceiling, the failed
probes with their notes (rewrite unclear notes), general QA and drift. Then start the server and the proxy
(see [vLLM](#vllm) and [The proxy](#the-proxy)), and publish:

```bash
pgs learn publish stores/acme --scope org:acme --servers http://10.0.0.5:8000
pgs learn key stores/acme --scope user:acme/support/ana --keys keys.yaml
pgs learn proxy stores/acme --keys keys.yaml --upstream http://10.0.0.5:8000 &
```

**Every week.** Pull the docs and refresh the source: unchanged chunks keep their notes, changed chunks
supersede the old ones (their old values become stale values), removed chunks retire. Until the next sleep
the proxy already injects the changed notes.

```bash
git -C acme-docs pull
pgs learn add stores/acme --scope org:acme --documents "**/*.md" --root acme-docs --approve
pgs learn inbox stores/acme --scope org:acme      # settle conflicts the refresh raised, if any
```

**Every night** (cron): `pgs learn sleep stores/acme --publish --budget-minutes 240`. The report of the
session shows the update and stale metrics on the changed values (the new price, not the old one), retention
of earlier facts and general QA; a failed gate keeps the notes in the prompt and names a remedy. Users'
"remember ..." notes consolidate into their own adapters the same night.

**When something is wrong.** `pgs learn rollback stores/acme --scope org:acme` serves the previous version
again and puts its notes back in the prompt; `pgs learn forget` removes a note for good.

## Store layout

```
stores/acme/
  learn_state.yaml          org, base, fingerprint (base revision, tokenizer, chat template, quantization), config
  LOCK                      one mutating command at a time
  SPOOL.lock                a short lock for the spools
  admin_queue.jsonl         promotions waiting for an admin
  published.json            what is published where: versions, composites, servers, canary results, history
  scopes/org/acme/
    sources.jsonl notes.jsonl facts.jsonl inbox.jsonl tombstones.jsonl
    spool.jsonl             signals the proxy captured while a job held LOCK (PII-screened)
    sessions/sNNNN/         questions, probes, near-miss and abstain prompts, config, losses,
                            report.json, report.md, diagnostics.jsonl, answers.jsonl
    pool/replay.jsonl       replay prompts only (no answers)
    adapters/vNNNN/         PEFT adapter + manifest.json
    LIVE
  scopes/team/acme/support/ ...
  composites/<scope>/<id>/  a stacked chain (sglang/: the same with SGLang's module names)
```

A store refuses to train on a base whose revision, tokenizer, chat template or quantization differs from the
one it was first trained on: adapters fit only the exact base.

## Configuration

`pgs learn init --config learn.yaml` stores a strict config (unknown options are errors); `pgs learn run`
takes `--section.option value` overrides.

```yaml
model:
  system_prompts: []                 # deployment system prompts, sampled into student and teacher rows
  chat_template_kwargs: {}
scopes:
  ranks: {org: 32, team: 16, user: 16}
adapter:
  alpha_over_rank: 2.0
notes:
  chunk_tokens: 400
  teacher_neighbours: 4
  relevance_threshold: 0.0           # BM25 score for a note to enter an anchor / near-miss teacher context
  wake_max_tokens: 4000              # notes a served prompt carries between sessions (0 = no limit)
signals:
  command_patterns: [remember, ricorda, ricordati, recuerda, souviens-toi, merk dir]
  max_note_chars: 1000
  max_notes_per_hour: 20             # captured by the proxy per key (0 = no limit)
pii:
  user_scope_personal: allow         # allow | redact | block
  shared_scope_personal: redact
generate:
  generator: self                    # self (the base model) | stub | a local model id
  questions_per_fact: 4
  probes_per_fact: 2
  nearmiss_per_fact: 2
  abstain_per_session: 20            # questions the notes do not answer (confabulation probes)
recipe:
  mode: distill                      # distill | sft
  divergence: fkl                    # fkl | rkl | jsd
  onpolicy_fraction: 0.5
  replay_teacher: notes              # notes | self
  anchor_teacher: notes              # notes | base | previous
  mix: {new: 0.45, replay: 0.25, nearmiss: 0.10, anchor: 0.20}
  passes: 4
  rows_per_step: 16
  learning_rate: 2.0e-4
  sampler: hf                        # hf | vllm_colocate (not available yet)
replay:
  visits_per_fact: 1
  max_prompts_per_fact: 4
eval:
  max_probes_per_session: 200        # acquisition and update probes
  max_retention_probes: 200
  max_ancestor_probes: 100
  general_qa_path: ""                # JSONL of {"q", "answers": [...]}; empty = built-in multilingual set
  drift_path: ""                     # JSONL of {"prompt"} or {"messages"}; empty = built-in prompts
  max_drift_prompts: 32
  max_new_tokens: 64
gates:
  calibrated: false                  # the defaults below are uncalibrated
  acquisition_min: 0.80
  update_min: 0.80
  stale_max: 0.05
  retention_max_net_loss: 3
  ancestor_retention_max_net_loss: 3
  general_qa_max_net_loss: 5
  leak_max_net: 2
  confab_max_net: 2
  meta_reference_max: 0.02
  drift_increment_max: 0.03
  drift_cap: 0.15
  parity_answer_max_net_loss: 1      # the parity canary of a publish
  parity_effect_corr_min: 0.9
  parity_k3_ratio_max: 1.5
  parity_k3_floor: 1.0e-4
serving:
  engine: vllm                       # vllm | sglang
  servers: []                        # every replica's base URL
  transport: api                     # api (load route) | resolver (vLLM plugin)
  base_name: ""                      # the base's name on the servers (default: the store's base)
  server_store: ""                   # the store's path as the servers see it
  drain_seconds: 60
  canary_probes: 16
  canary_max_new_tokens: 48
  timeout: 120.0
  api_key_env: PGS_LEARN_SERVER_KEY  # environment variable with the servers' --api-key
sleep:
  budget_minutes: 0                  # 0 = no limit
  scope_budget_minutes: 0
  children_follow: gated             # gated | none
```

## Limits

- Experimental; recipe defaults are untested hypotheses at scale, and the gate thresholds are uncalibrated.
- The CPU tests check the evaluation's mechanics on tiny random models, not that its thresholds are right.
- Serving is tested on CPU against fake servers; the GPU integration tests (`tests/test_learn_gpu.py`: vLLM
  load and versioned swap on a hybrid and a dense model, prefix-cache behaviour, HF + PEFT parity) have not
  run yet, and the canary thresholds are uncalibrated like the gates.
- One vLLM API server process per server; several front-end processes do not share runtime-loaded names.
- On-policy sampling uses HF `generate` (slow); the colocated vLLM sampler is an interface only.
- MLP-only LoRA; no MoE expert adapters.
- Not a replacement for retrieval over large corpora.

## Related work

- Context distillation: Snell et al., arXiv 2209.15189.
- Generalized knowledge distillation (on-policy, GKD): Agarwal et al., arXiv 2306.13649.
- Knowledge editing and where facts live: ROME (arXiv 2202.05262), MEMIT (arXiv 2210.07229), feed-forward
  layers as key-value memories (Geva et al., arXiv 2012.14913).
- Retrieval versus fine-tuning for new knowledge: Ovadia et al., arXiv 2312.05934; new knowledge and
  hallucination: Gekhman et al., arXiv 2405.05904.
- Serving many adapters on one base: Punica (arXiv 2310.18547), S-LoRA (arXiv 2311.03285); LoRA forgets less
  than full fine-tuning: Biderman et al., arXiv 2405.09673.
