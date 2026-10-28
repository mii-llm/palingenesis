# Continual Learning (experimental)

!!! warning "Experimental"
    `pgs learn` lives on a feature branch. Its interfaces and defaults may change, and every recipe default
    is a hypothesis until a benchmark settles it. Evaluation gates, publishing and serving are not
    implemented yet: a trained adapter is not checked before you make it live.

`palingenesis.learn` keeps what a deployed model must know per **scope** (an organization, a team, a person)
and consolidates it into one small LoRA adapter per scope.

```bash
pip install 'palingenesis[learn]'
pgs learn init stores/acme --base Qwen/Qwen3.5-4B --org acme
pgs learn add stores/acme --scope org:acme --documents "docs/**/*.md" --approve
pgs learn run stores/acme --scope org:acme --live
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
| personal | emails, phone numbers, Italian codice fiscale, US social security numbers | kept (default) | redacted (default) |

`pii.user_scope_personal` and `pii.shared_scope_personal` take `allow`, `redact` or `block` (held for review).
The regex detectors are a baseline (no postal addresses, no national ids beyond the codice fiscale and US
SSN): pass your own detectors (any object with `find(text)`) to `PIIFilter`.

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

# Train the scope's adapter on its new facts; --live makes the version live
pgs learn run stores/acme --scope org:acme --live
pgs learn run stores/acme --scope user:acme/support/ana --live --recipe.passes 2

# Next week: refreshed documents supersede the changed chunks only
pgs learn add stores/acme --scope org:acme --documents "**/*.md" --root acme-docs --approve
pgs learn run stores/acme --scope org:acme --live

# Forget a note everywhere
pgs learn forget stores/acme --scope user:acme/support/ana n_8c1d...
```

The same from Python:

```python
from palingenesis.learn import Learner

learner = Learner.init("stores/acme", base="Qwen/Qwen3.5-4B", org="acme")
learner.scope("org:acme").add_documents("**/*.md", root="acme-docs", approve=True)
learner.intake("user:acme/support/ana", [{"role": "user", "content": "ricorda che preferisco risposte brevi"}])
result = learner.run("org:acme")
learner.set_live("org:acme", result.version)
composite = learner.stack("user:acme/support/ana")   # org + team + user as one PEFT adapter
messages = learner.render_notes("user:acme/support/ana", messages)   # notes not consolidated yet
report = learner.forget("user:acme/support/ana", ["n_8c1d..."])
assert report.ok
```

Between sessions, `render_notes` puts the chain's unconsolidated notes into the system turn as quoted
reference data, in the same format the training teacher sees.

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

Each session writes `sessions/sNNNN/` (questions, probes, near-miss prompts, config, losses) and
`adapters/vNNNN/` (a PEFT adapter and a manifest listing the notes active during training and the parent
versions it was trained on). A failed session leaves no files behind.

!!! note "What the CPU tests show"
    The test suite runs the whole loop on tiny random models. It checks mechanics (rows, teachers, frozen
    ancestors, stacking, crash safety), not that facts are learned.

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
   characters or more). The command exits with 1 and lists the files if any still hold one.

## Store layout

```
stores/acme/
  learn_state.yaml          org, base, fingerprint (base revision, tokenizer, chat template, quantization), config
  LOCK                      one mutating command at a time
  admin_queue.jsonl         promotions waiting for an admin
  scopes/org/acme/
    sources.jsonl notes.jsonl facts.jsonl inbox.jsonl tombstones.jsonl
    sessions/sNNNN/         questions, probes, near-miss prompts, config, losses
    pool/replay.jsonl       replay prompts only (no answers)
    adapters/vNNNN/         PEFT adapter + manifest.json
    LIVE
  scopes/team/acme/support/ ...
  composites/<scope>/<id>/  a stacked chain
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
signals:
  command_patterns: [remember, ricorda, ricordati, recuerda, souviens-toi, merk dir]
  max_note_chars: 1000
pii:
  user_scope_personal: allow         # allow | redact | block
  shared_scope_personal: redact
generate:
  generator: self                    # self (the base model) | stub | a local model id
  questions_per_fact: 4
  probes_per_fact: 2
  nearmiss_per_fact: 2
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
```

## Limits

- Experimental; defaults are untested hypotheses at scale.
- No evaluation gates, reports, publishing or serving integration yet; `--live` is ungated.
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
