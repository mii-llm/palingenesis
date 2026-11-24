# Data Preparation

*`pgs prepare` cleans a dataset before training: it removes the rows training would drop and exact duplicates, holds out an eval set that shares no prompt with the training data, caps sources, and reports every row it removed and why. It can also score every row with the model; nothing selected by that score has beaten a random subset.*

---

!!! abstract "What this page covers"
    - **Cleaning** with `pgs prepare`: [what it does](#what-preparation-does), [the pipeline](#the-pipeline), [scoring with the model](#scoring-with-the-model-experimental) (experimental) and [what the measurements say](#what-the-measurements-say) (short version: nothing selected by the score beat a random subset).
    - **Mixing sources** (`data.sources`): [multi-source mixing](#multi-source-mixing).
    - **What gets loss**: [SFT masking](#what-gets-loss-sft-masking), [reasoning fields and tags](#reasoning-in-the-data-fields-baked-blocks-tags); agent traces and tool calls are in the [agentic guide](agentic-training.md).
    - **Run length, streaming and the [pre-tokenized cache](#pre-tokenized-cache)**.
    - **[Validation data](#validation-data)**, including per-source eval losses (`eval_sources`).
    - **Raw text** (`mode: pretrain`, continued pretraining on a corpus): [Pretraining and Continued Pretraining](pretraining.md).

    Check the masking of any config with `pgs inspect --config <cfg> --num_samples 3` before training. Every key is in the [configuration reference](../reference/configuration.md); every command in the [CLI reference](../reference/cli.md).

## What preparation does

By default `pgs prepare` needs only the tokenizer, no GPU and no forward pass:

1. **Render** every row with the trainer's own pipeline (chat template, tools, truncation at turn boundaries, masking). Rows training would drop are removed and named: `nothing_to_train_on` (no messages, or no assistant turn that gets loss), `no_trained_turn_fits` (the first trained turn already exceeds `data.max_seq_length`), `unrenderable` (the template rejects the row). Rows that are kept but cut to `max_seq_length` are counted. Rows are rendered in parallel processes (`num_workers`, default: the CPUs available, at most 8); the output is the same as in one process.
2. **Deduplicate**: exact duplicates (same messages and tool definitions, whitespace runs collapsed) keep their first row. `max_per_prompt: N` also keeps at most N rows that share a system prompt and first user turn, for example several trajectories of one task.
3. **Hold out** (`eval_holdout: N`): about N rows, drawn as whole prompts, so the eval set never holds a task the model trained on under another answer.
4. **Cap** (`group_field: source` with `max_per_group` / `max_tokens_per_group`): at most that many rows or tokens per source, drawn at random inside the source.
5. **Select** (`budget` rows, `token_budget` tokens): a seeded random subset.
6. **Write** `scored_data.parquet`, `prepared_meta.json` and `dropped.jsonl`.

What you get back:

| File | Content |
|------|---------|
| `scored_data.parquet` | The kept rows, with `_row` (index in the raw data), `_score_length` (tokens as training sees the row) and `_score_response_token_count` (tokens that get loss). |
| `eval_data.parquet` | The held-out rows (`eval_holdout > 0`). |
| `prepared_meta.json` | Rows loaded, removed per reason, cut to `max_seq_length`; total and trained tokens, length percentiles; per-group rows and tokens before and after caps and selection; every setting used. |
| `dropped.jsonl` | One line per removed row: `row`, `reason`, and `duplicate_of` or `group`. |

Read the report before training. A large `no_trained_turn_fits` or cut count means `max_seq_length` is too small for the data; a large `nothing_to_train_on` usually means the wrong `messages_field`; many duplicates in a public mixture are common.

### When not to use it

- **To pick the "best" samples.** There is no setting here that finds them. See [Scoring with the model](#scoring-with-the-model-experimental).
- **To fix forgetting or a regression.** In the [measurements](#what-the-measurements-say) the source of the responses and the learning rate moved results; selection did not.
- **To judge correctness.** Nothing here checks that an answer is right or that a trajectory solved its task. Filter on outcomes (tests passed, verifier reward) before preparing; for tool calls, `validate_data.validate_dataset` checks names and arguments against the declared tools.
- **For `data.sources` mixes** use `pgs prepare-multi` (below), or prepare each source and point `data.sources` at the outputs.

---

## The pipeline

=== "Config-driven (recommended)"

    One YAML drives everything. `pgs prepare` reads the **same config** you train with: the tokenizer is `model.name_or_path`'s, the raw data is `data.dataset`, rows are rendered and masked with the `data` options, and the `preprocess:` section controls cleaning and selection:

    ```bash
    pgs prepare --config configs/qwen35_4b/a100_80gb.yaml
    ```

    Then train — with `preprocess.enabled: true`, training automatically picks up the prepared parquet instead of the raw dataset:

    ```bash
    pgs train --config configs/qwen35_4b/a100_80gb.yaml --preprocess.enabled true
    ```

    The same `--section.field` overrides work here too:

    ```bash
    pgs prepare --config cfg.yaml --preprocess.eval_holdout 200 --preprocess.max_per_prompt 2
    ```

=== "Standalone flags"

    No config file needed:

    ```bash
    pgs prepare \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data my_data.jsonl \
        --output prepared/ \
        --eval_holdout 200 \
        --group_field source --max_tokens_per_group 20000000
    ```

Lengths and masks are the trainer's: a row's `_score_length` is the number of tokens it will occupy in a batch, and its trained tokens are the ones that get loss with your `data` options (`train_on_reasoning`, `include_observations`, `last_turn_only`, `think_tags`, `chat_template_kwargs`, per-message `loss: false`).

Cost: rendering and tokenizing on one CPU core. Measured on a laptop: under 20 ms per row for agent traces of about 2,000 tokens, most of them cut to fit, so about half an hour per 100,000 such rows; longer rows cost proportionally more. The rows are held in memory.

---

## One config, closed loop

The preparation → training loop is fully wired through a single YAML. There is no way for the scoring model and the training model to drift apart, because they are the same field.

<figure class="pg-figure">
<div class="pg-figure__canvas">
<svg class="pg-fig" viewBox="0 0 700 170" role="img" aria-labelledby="fig-prep-title fig-prep-desc">
<title id="fig-prep-title">Two ways from a config to training: prepare once, or prepare inside pgs train</title>
<desc id="fig-prep-desc">The same config.yaml, with model, data and preprocess sections, feeds two paths. pgs prepare --config writes a prepared directory with scored_data.parquet, prepared_meta.json and dropped.jsonl, which training then reads. Or pgs train --config with preprocess.enabled true trains from the prepared output.</desc>
<defs><marker id="fig-prep-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head" d="M0,0 L10,5 L0,10 z"/></marker><marker id="fig-prep-arrow-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="head-accent" d="M0,0 L10,5 L0,10 z"/></marker></defs>
<rect class="box" x="10" y="56" width="170" height="62" rx="4"/>
<text class="" x="22" y="76">config.yaml</text>
<text class="t-muted t-small" x="22" y="92">model + data + preprocess</text>
<rect class="box" x="300" y="6" width="190" height="76" rx="4"/>
<text class="" x="312" y="30">prepared/</text>
<text class="t-muted t-small t-mono" x="312" y="46">scored_data.parquet</text>
<text class="t-muted t-small t-mono" x="312" y="60">prepared_meta.json</text>
<text class="t-muted t-small t-mono" x="312" y="74">dropped.jsonl</text>
<rect class="box-accent" x="560" y="56" width="130" height="62" rx="4"/>
<text class="" x="572" y="76">training</text>
<text class="t-muted t-small" x="572" y="92">pgs train</text>
<path class="line" d="M180,72 C230,72 240,41 298,41" marker-end="url(#fig-prep-arrow)"/>
<text class="t-mono t-muted" x="196" y="22">pgs prepare</text><text class="t-mono t-muted" x="196" y="36">--config</text>
<path class="line" d="M490,41 C530,41 530,72 558,74" marker-end="url(#fig-prep-arrow)"/>
<path class="line" d="M180,104 C300,140 450,140 558,104" marker-end="url(#fig-prep-arrow)"/>
<text class="t-mono t-muted" x="370" y="146" text-anchor="middle">pgs train --config  (preprocess.enabled: true)</text>
</svg>
</div>
<figcaption>Preparation reads the same config as training: the same tokenizer, chat template and truncation clean the data that training then reads.</figcaption>
</figure>

```yaml title="config.yaml (excerpt)"
model:
  name_or_path: Qwen/Qwen3.5-4B     # its tokenizer and chat template render the rows

data:
  dataset: your-org/agentic-traces  # the RAW dataset to prepare
  max_seq_length: 16384             # same truncation in prepare and training

preprocess:
  enabled: true                     # training uses the prepared output
  output_dir: ./prepared/qwen35_4b
  eval_holdout: 200                 # held-out eval set, no prompt shared with training
  max_per_prompt: 2                 # at most 2 trajectories per task
  group_field: source               # row column naming the source
  max_tokens_per_group: 20000000    # no source above 20M tokens
  budget: 10000                     # rows to keep (0 = all); token_budget counts tokens instead
```

!!! tip "Held-out eval set"
    Set `eval_holdout: N` and prepare reserves about N rows from the cleaned pool, before caps and selection, into `eval_data.parquet`. Rows that share a prompt stay on one side, and exact duplicates are already gone, so nothing in the eval set was trained on. If `data.eval_dataset` is left empty, training picks this file up automatically, so `eval/loss`, `eval/ppl` and `eval/gap` measure generalization on your own distribution. When a few prompts hold most rows the set can come out smaller than N (a warning says so).

!!! tip "Provenance travels with the data"
    Every prepare run writes `prepared_meta.json` next to the data, and training logs it at startup, so every run records which preparation produced its data.

!!! warning "No silent fallback"
    If `preprocess.enabled: true` but nothing has been prepared yet, training **fails immediately** with the exact command to run. It will never silently fall back to the raw dataset.

**Parquet is the default** because it preserves sample order, loads far faster than JSONL, and is consumed directly by the training data loader (you can point `data.dataset` at any `.parquet` file or prepared directory manually, too). If your samples have a schema Arrow can't unify, the writer falls back to JSONL automatically.

---

## Scoring with the model (experimental)

`preprocess.score: true` (or `--score`) adds one forward pass with the model you will fine-tune and records, per row, the perplexity of the tokens training puts loss on (`_score_response_ppl`, `_score_response_nll`) and a familiarity bucket (bottom 25% / middle / top 25% of *your* dataset under *your* model). A low value means the model already writes this response almost verbatim; a high value means the response is far from what the model would write. It is a property of the (model, response) pair, not of the problem: a correct but unusual solution to an easy problem scores "unfamiliar".

Scoring also runs when an option needs it: `min_ppl` / `max_ppl` above 0, `hes`, or any `strategy` other than `random`.

**What the evidence says.** Selecting or ordering training data by this score has not beaten a random subset of the same size:

- Our [measurements](#what-the-measurements-say) on Qwen3.5-0.8B: no strategy beat random, the J-shaped mix was worse, and the score had no correlation with whether the model solves the problem.
- arXiv:2410.09335 (Oct 2024) re-ran loss-, perplexity- and gradient-based selectors on pools of 0.4M–1M instruction examples for Llama3-8B and Qwen2-7B: most fell inside the range of five random draws, and perplexity scores tracked response length. arXiv:2410.15225 (Oct 2024) and arXiv:2503.01807 (Mar 2025, pools up to 5.8M) report that selectors often fail to beat random across sources, budgets and pool sizes.
- Easy-to-hard ordering: arXiv:2012.03107 found curricula help little outside short training budgets and noisy labels; the LLM instruction-tuning papers that report gains report small ones (about one point) and none orders by the model's own perplexity.
- HES (arXiv:2605.22389, May 2026) reports top-20% selection matching the full set on long math/STEM reasoning for one base model; its authors did not test agentic or tool-use data.

So use the score to **look** at a dataset (per-source perplexity in `prepared_meta.json`, the tail of very high values, which is often broken or wrong-language text), not to choose what to train on, unless you have compared against `strategy: random` on your own eval battery.

| Option | What it does | Status |
|--------|--------------|--------|
| `min_ppl` / `max_ppl` | Drop rows whose response perplexity is below / above the bound (0 = off, the default) | Absolute perplexity depends on model, domain and language; short tool calls and templated answers score very low. Read the distribution first. |
| `strategy: optimal` | J-shaped mix: 20% familiar / 50% typical / 25% unfamiliar / 5% most unfamiliar above 10K samples; 25/50/20/5 from 2K to 10K; 35/50/15/0 below 2K | A heuristic of ours. **Measured below `random`.** |
| `strategy: flow` | The lowest-perplexity samples | Named after FLOW (arXiv:2502.02797), which *re-weights* the loss by exp(−loss/τ) to reduce forgetting and reports slightly lower target accuracy; palingenesis records that weight as `_score_flow_weight`, training does not use it. Measured: no gain over random. |
| `strategy: curriculum` | A random subset stored familiar→unfamiliar; training keeps that order (no shuffling, no length grouping) in every epoch | Not measured. Batches stop being a random draw of the data, and the run ends on its highest-perplexity rows. |
| `strategy: balanced` / `medium_focus` / `hard_focus` | Equal thirds / closest to the median / highest perplexity | Not measured. |
| `hes` | Also writes `_score_hes` (sum of the top `hes_top_k_pct`% token entropies of the response, from the same forward) and `_score_hes_mean` | No strategy reads it. The sum grows with response length; compare rows of different length with the mean. |

Every strategy other than `random` logs a warning. Within a bucket samples are drawn at random, never in file order. Cost of scoring: 20,000 math conversations (10M tokens) took 7 minutes for Qwen3.5-0.8B on one A100; time grows with model size and token count, so 100,000 traces of 8K tokens are hours on a 4B model. A run that stops halfway starts over.

## What the measurements say

One controlled study, so read it as evidence for one setting, not as a law: **Qwen3.5-0.8B** (the post-trained hybrid checkpoint, non-thinking), fine-tuned on **4,000 math conversations** for 2 epochs (LR 1e-5 cosine, 32 conversations/step, full fine-tuning); only the data changes between runs. Evaluated with sampling on GSM8K, MATH-500, IFEval, MMLU-Pro, HumanEval+, MBPP+ and a chat sanity check; differences are paired per item against the untouched model.

| Data (4,000 conversations) | GSM8K | MATH-500 | IFEval | MMLU-Pro | HumanEval+ |
|---|---|---|---|---|---|
| untouched model | 56.3 | 40.8 | 56.7 | 31.8 | 28.7 |
| random subset, dataset (GPT-4o) solutions | −1.7 | −6.3 | −23.2 | −1.0 | −10.4 |
| `optimal` (J-mix) subset | −4.2 | −8.3 | −25.9 | −1.2 | −14.3 |
| most familiar 20% (≈ `flow`) | +0.5 | −8.4 | −23.1 | +1.2 | −8.8 |
| same prompts, dataset solutions | −0.5 | −6.8 | −28.5 | −1.6 | −5.8 |
| same prompts, **the model's own verified solutions** | +1.4 | +1.8 | −17.5 | +1.1 | −3.0 |
| random subset at LR 2e-6 instead of 1e-5 | −3.7 | −3.8 | −11.3 | −3.1 | −7.9 |
| random subset + 4,000 general chat conversations | −3.4 | −6.7 | −18.9 | −3.4 | −11.9 |
| random subset + the same 4,000 chat prompts **answered by the model itself** | −2.9 | −6.6 | −10.2 | +2.4 | −1.8 |

Standard errors of a single cell are about 1.3 (GSM8K, MMLU-Pro), 2.1 (MATH-500), 2.4 (IFEval) and 3.5 (HumanEval+); two training seeds of the same data differ by up to 3 points. Every row averages two seeds except the LR 2e-6 row and the last row (one seed each).

What this supports:

- **Selection by perplexity did not help.** No strategy beat a random subset on the target skill, and `optimal` was worse than random on GSM8K (−2.4 ± 1.1) and HumanEval+ (−4.0 ± 1.8), consistently across both seeds. Hence `random` is the default and scoring is off unless asked for.
- **The score is not difficulty.** On 20,000 NuminaMath problems, the Spearman correlation between the model's pass rate (4 samples) and the response perplexity of the reference solution was 0.009. Selecting the problems the model solves sometimes (pass rate between 0 and 1) did not help either.
- **Where the responses come from matters more than which samples you pick.** On identical prompts, the model's own verified solutions (sample 4 answers, keep a correct one) beat the dataset's solutions by +8.6 ± 1.7 on MATH-500, +11.0 ± 1.8 on IFEval and +2.7 ± 1.1 on MMLU-Pro (two seeds each). They did not *raise* math above the untouched model, though: at this scale they preserve the skill rather than improve it. The dataset solutions are much shorter than what the model writes, and the model learns that: its MATH-500 answers shrink from ~1,250 to ~460 tokens and its accuracy drops. The price: the model keeps its habit of long answers that sometimes never terminate (chat sanity −7.5 ± 1.6 and MBPP+ −3.4 ± 1.8 vs dataset solutions).
- **Forgetting is the main effect, and data selection does not fix it.** Every run lost 11–29 points of IFEval. A 5× lower learning rate halved the IFEval loss (−11.3 vs −23.2) but still gave no math gain; mixing in an equal amount of general chat data recovered only +4.3 ± 1.7, because that chat data itself costs IFEval when trained alone (−20.9).
- **Replay the model's own answers, not a dataset's.** Answering the same 4,000 general prompts with the model itself (sampled once, unfinished answers dropped) and mixing them in instead of the dataset's answers kept +8.7 ± 2.3 IFEval, +6.2 ± 1.4 MMLU-Pro and +9.1 ± 3.4 HumanEval+ more (one seed). It does not protect the target skill: that depends on where the target responses come from (above). To do this in palingenesis, generate the answers with vLLM and add them as a second `data.sources` entry; `pretrain_replay_dataset` replays raw text, which we did not test.

What it does not show: other model sizes, thinking mode, base (non-instruct) models, larger budgets, or skills other than math. On base models the cited papers find harder data more useful as the budget grows (2605.12906). Before relying on any selection, run the same comparison against `random` on your own model and eval battery.

---

## Semantic packing (TFP)

TFP (Threshold Filtering Packing, Dong et al. 2024) orders samples so that related but not identical conversations end up adjacent, then packed into the same sequence, where each one serves as an implicit demonstration for the next. That mechanism needs packed documents to attend to each other. palingenesis keeps every packed conversation isolated (a document never sees the one before it; see [packing](../getting-started/first-training.md#packing-when-to-use-it)), so a TFP ordering changes nothing the model sees and is not part of the pipeline. `palingenesis.tfp.compute_tfp_ordering` remains available as a standalone utility for data exploration.

---

## Multi-source mixing

Real projects have multiple data sources: agentic traces, general instruction-following, code, reasoning.

```yaml title="sources.yaml"
- dataset: ./agentic_traces.jsonl
  name: agentic
  weight: 0.65

- dataset: ./general_instruct.jsonl
  name: general
  weight: 0.20

- dataset: ./code_verified.jsonl
  name: code
  weight: 0.15
```

```bash
pgs prepare-multi --model Qwen/Qwen3.5-4B --sources sources.yaml --output prepared/
```

Each source is prepared on its own (same cleaning as `pgs prepare`, duplicates removed inside a source, not across sources) into `prepared/<name>/scored_data.jsonl`, and `prepared/manifest.json` lists per source the rows, tokens, what cleaning removed, and what the weights amount to: `normalized_weight` (share of examples) and `token_share` (share of tokens). The two differ whenever sources differ in length, which is the usual case with agentic traces next to chat data. Point `data.sources` at the per-source files to train on them. Raw-text sources (`mode: pretrain`) are cleaned as documents instead (exact and near-duplicates, n-gram decontamination: [Preparing the data](pretraining.md#preparing-the-data)). `--score`, `--min_ppl`, `--max_ppl`, `--strategy` and `--hes` behave as in `pgs prepare` and are [experimental](#scoring-with-the-model-experimental).

How the mix behaves in training (`data.sources`):

- **An epoch is as many rows as the sources hold together**, drawn by weight (`data.mix_epoch: total`). A conversation is one row and one training sequence; a raw-text document split into several chunks counts once, when its last chunk is drawn. `weight` is a relative **per-sequence** sampling probability, not a token share: a source of long conversations contributes more tokens than its weight suggests. `data.mix_weights: tokens` makes the weights shares of the tokens instead (each source is drawn in proportion to its weight divided by the mean length of its sequences; every source then needs a weight).
- **A source without a weight gets its share of the rows**, so a mix without weights is one shuffled concatenation of its sources, one pass per epoch. A source weighted above its share is repeated (a new order every pass); one weighted below it contributes a fresh random subset every epoch.
- The log gives every source's share, rows and passes per epoch at startup, e.g. `mix math: 50.0% of sequences, 20,000 rows, 2.50 passes/epoch (upsampled)` (`of rows` when no source has a weight, `of tokens` with `mix_weights: tokens`).
- `data.mix_epoch: first_exhausted` ends the epoch when a source runs out instead (the former rule; also what happens when a source's size cannot be read from its metadata). Replay (`pretrain_replay_dataset`) is drawn alongside one pass over the target data's rows.

!!! warning "`data.msft_tracking` has no effect yet"
    `msft.AdaptiveSourceTracker` (per-source weight decay when a source's validation loss rises, after mSFT, arXiv:2603.21606) exists but is not wired into the training loop. Setting `msft_tracking: true` logs a warning and the weights stay fixed. Watch per-source losses with `eval_sources` instead.

---

## What gets loss (SFT masking)

A sample is one of two shapes, and each is scored differently:

- **`{"text": "..."}` (pretrain / raw LM):** *every* token gets loss. There is no masking — the whole string is next-token prediction. Selected per source with `mode: pretrain` + `text_field`.
- **`{"messages": [...]}` (SFT / chat):** only **assistant** tokens get loss; system and user turns are masked out. Selected with `mode: sft` (the default) + `messages_field`.

!!! warning "Don't feed a rendered conversation as `text`"
    A full ChatML string (`<|im_start|>user…assistant…`) shoved into a `text` field trains on the *entire prompt* — question included — because pretrain mode masks nothing. For "loss only on the answer," use `messages` + `mode: sft`.

### How assistant tokens are located

Palingenesis masks purely from the model's own chat template, two ways:

1. **Fast path** — templates with a `{% generation %}` span expose Hugging Face's native assistant mask; that mask defines the trained tokens exactly.
2. **Turn-marker path** — templates without a generation span (Qwen3/3.5, Llama 3, ...). The assistant header and end-of-turn marker are derived from the template itself by rendering probe conversations; each assistant turn is then everything between its header and its end-of-turn token in the rendered string, found via offset mapping. That includes text the template renders from other fields, such as tool calls, and the end-of-turn token (the model learns to stop). Only the final render is used, with **no prefix-consistency assumption**, so it stays correct for templates that rewrite history — e.g. Qwen3.x dropping `<think>` from past turns. Templates without special-token markers fall back to locating each turn's text.

Per-message flags (`"loss": false`) and the `tools` field are honoured on both paths (see the [agentic guide](agentic-training.md)).

Both paths honor the same two knobs, identically:

| Option | Effect |
|--------|--------|
| `train_on_reasoning` (default `true`) | `true`: loss on the `<think>` block **and** the answer (distils reasoning). `false`: loss only on the post-`</think>` answer — the reasoning is stripped even when the template's generation span encloses it. |
| `last_turn_only` (default `false`) | Loss only on the **final** assistant turn; earlier assistant turns are masked. Use when earlier turns are a fixed context you must not fit — e.g. n-shot MCQA exemplar answers. No-op for single-turn data. |

An **empty** `<think>\n\n</think>` scaffold (as Qwen fast-format emits) carries no reasoning, so nothing inside it is trained regardless of `train_on_reasoning`; loss lands on the answer + terminator.

Both are set globally under `data:`. `last_turn_only` can additionally be overridden per source (in `sources` and `eval_sources` entries); `train_on_reasoning` is global-only.

### Reasoning in the data: fields, baked blocks, tags

An assistant turn's reasoning can come as a field (`reasoning`, `reasoning_content` or `think`) or baked into the content, as a block that **opens** the message. A baked block is moved into the reasoning field, so the template renders it exactly once. Templates that put an empty `<think>\n\n</think>\n\n` in front of turns without reasoning never add one before the baked block. If a row has both a field and a baked block, the field wins. Think tags later in the answer are text: they are rendered as written and trained. Templates that do not render reasoning at all (e.g. SmolLM2) get the content back verbatim.

| Option | Effect |
|--------|--------|
| `think_tags` (default: the template's) | Delimiters of the baked blocks, e.g. `["[THINK]", "[/THINK]"]`. By default they are read off the chat template (`<think></think>` for Qwen3.x, GLM, MiniMax; `[THINK][/THINK]` for Magistral-style templates), else `<think></think>`. Masking always uses the template's own tags, so data written with another model's tags (e.g. `◁think▷◁/think▷`) is converted to this model's format. |
| `chat_template_kwargs` (default `{}`) | Template kwargs for every row, e.g. `{enable_thinking: true}`. A row's own `chat_template_kwargs` column overrides them key by key, so thinking and non-thinking rows mix in one dataset. |

Both can be overridden per source (`sources` entries). A source's `chat_template_kwargs` is merged over the global kwargs. Agent-trace distillation sources take `think_tags` too.

```yaml
data:
  think_tags: ["<think>", "</think>"]        # optional: the student template's by default
  chat_template_kwargs: {enable_thinking: true}
  sources:
    - {dataset: data/reasoning.parquet, weight: 0.7}
    - {dataset: data/chat.parquet, weight: 0.3, chat_template_kwargs: {enable_thinking: false}}
    - {dataset: data/kimi_traces.parquet, weight: 0.2, think_tags: ["◁think▷", "◁/think▷"]}
```

---

## Run length and streaming

Training starts at once, streaming included: the LR schedule's horizon (the number of optimizer steps of an epoch-based run) is **estimated**, not counted by a pass over the data.

- **Rows** come from metadata: a parquet footer, a JSONL's newlines, an in-memory dataset's length, a Hub dataset's published split sizes.
- **What a row yields** (examples after filters, their tokens, packed sequences, kept preference pairs) is measured on a uniform random sample of rows run through the real pipeline, mixture weights included.
- **Stratified by row size.** The scan that counts a JSONL's rows also records every row's size (Arrow data: string sizes, computed column-wise), and sizes predict most of a row's fate: whether it fits `max_seq_length`, how many tokens it holds. The sample is used per size bin, weighted by that bin's share of *all* rows, so length-driven effects (the long tail that filters drop) are known for every row; only what length does not explain is left to sampling.
- **Sequential.** Rows are drawn in rounds until the reported error reaches 0.5% (`PALINGENESIS_SIZE_PRECISION`; 2,000 rows a round, `PALINGENESIS_SIZE_SAMPLE`), without sampling more than a quarter of a dataset; a dataset smaller than a round is read whole (exact).
- **Packing** is sized as $\text{tokens} / (\texttt{max\_seq\_length} \times \text{fill rate})$, the fill rate being the packer's: the tokens from the stratified estimate, only the fill rate (a stable ratio, ±0.03%) from running the real packer on a synthetic stream.

Measured: 60,000 conversations with packing, 1,895 micro-batches estimated in 3 s against 1,894 counted in 28 s (reported ±0.1%). Preference pairs with 12–30% dropped: errors of about ±0.5% (1 sigma, and the reported error matches it), against ±1.2% for a plain random sample. The log reports the estimate and its error, and at each epoch's end how many micro-batches it actually had. If the data outlasts the estimate the LR stays at its floor; if it ends a little sooner the decay stops a little early. `train.exact_steps: true` scans the whole pipeline instead; `train.max_steps` skips both.

`data.streaming: true` reads local JSONL and parquet files as many shards (byte ranges, row groups) that ranks and DataLoader workers read in parallel: without that, `datasets` streams a single file through one worker while the others stop, and every rank reads the whole file. Write parquet with several row groups (`pyarrow` `row_group_size`) so it can be split. A Hub dataset without published split sizes cannot be sized from metadata: set `train.max_steps`.

With several GPUs, each rank packs its own shard, so ranks can reach the end of an epoch a micro-batch apart; they agree before every micro-batch (on a CPU group, no GPU sync) and end the epoch together.

---

## Pre-tokenized cache

By default every sequence is tokenized, masked, mixed and packed **on the fly**, every epoch, in a new order each epoch. `pretokenize` runs the whole assembly **once**, dumps the final tensors to disk, and on every later run loads them directly (and reshuffles them every epoch after the first).

```yaml
data:
  pretokenize: true
  pretokenize_path: ./pretokenized   # train.parquet + pretokenized_meta.json
```

What you get:

- **No per-step tokenization** — the cached rows are the final `input_ids` / `labels` / `attention_mask` (plus `position_ids` when packed). The trainer just reads and collates them. With `packing: false`, length-grouped batching (`length_group_buffer`) is still re-applied at load time, so the cache keeps the pad-token throughput win.
- **Exact step count for free** — the cache's row count is its number of training sequences, read from the parquet footer.
- **Automatic invalidation** — a fingerprint over the tokenizer, chat template, `max_seq_length`, `packing`, every source (path + size + mtime + weight + mode + fields + `last_turn_only`), `train_on_reasoning`, `turn_scaling`, `include_observations`, `seed` and replay is stored in `pretokenized_meta.json`. Change any of them and the cache is rebuilt — you can never silently train on a stale tokenization.

The cache is a **static** stream, so `msft_tracking` (which adjusts per-source weights *during* training) is rejected at validation time with a clear error. Disable one of the two to proceed. Under multi-GPU, rank 0 builds the cache once and the other ranks wait on a barrier, then each rank reads a disjoint shard.

---

## Validation data

Always, always, always have validation data. The simplest form is a single held-out set:

```yaml
data:
  eval_dataset: my_data.jsonl
  eval_split: test
  eval_samples: 200
  eval_every: 50
```

Without it:

- No best-model tracking (you get the last checkpoint, which may be overtrained)
- No early stopping signal
- No MSFT source-level monitoring
- No RL-readiness entropy tracking

With it: palingenesis continuously evaluates and saves the best checkpoint. The cost is negligible (200 samples, no gradient, every 50 steps ≈ 2 seconds of overhead per hour of training).

### Per-capability eval (`eval_sources`)

A single mixed `eval_dataset` gets token-dominated by whichever source has the longest sequences, so one number hides per-capability regressions. `eval_sources` scores each source **independently** and combines them into a weighted composite that drives best-model tracking (logged as `eval/<name>/loss` + `eval/loss`):

```yaml
data:
  eval_every: 50
  eval_sources:
    # raw-text language modeling — all-token CE/perplexity, no chat template
    - name: lm
      dataset: ./eval/heldout_docs.jsonl
      split: test
      mode: pretrain          # {"text": "..."}
      text_field: text
      weight: 0.4
      samples: 200
    # chat/MCQA proxy — assistant-only CE
    - name: mcqa
      dataset: ./eval/mcqa_heldout.jsonl
      split: test
      mode: sft               # {"messages": [...]}
      messages_field: messages
      last_turn_only: true    # score only the final answer (n-shot prefix ignored)
      weight: 0.6
      samples: 100
```

Every evaluation also prints the per-source losses on the plain log line, after the composite, so forgetting on one source shows without a tracker:

```
step=… eval=<composite> eval_sources=lm:<loss>,mcqa:<loss>
```

The baseline evaluation before the first step is printed the same way, marked `(before training)`.

Per-source keys: `name`, `dataset` (a local `.jsonl`, `.json` or `.parquet` file, a `pgs prepare` output directory, or a Hub dataset with its `split`), `weight` (composite importance), `samples` (subset size), `regression_floor` (optional alarm), and **`mode`**. A source that cannot be read, has an unknown mode or gives no usable sample stops the run with a config error: a configured evaluation set is never skipped.

- **`mode: pretrain`** (+ `text_field`, `append_eos`): raw-text, all-token CE/ppl, no chat template, the end-of-document token included as in training. Matches how CPT actually trains, so the number is a true next-token perplexity. Use it for held-out LM text.
- **`mode: sft`** (default, + `messages_field`, optional `last_turn_only`): chat-templated, assistant-only CE. Use for genuine chat/MCQA tasks.

!!! warning "Measure raw LM as `text`, not as a fake assistant turn"
    Wrapping plain LM text in an `{"role": "assistant"}` message conditions perplexity on the chat-template scaffolding and no longer measures raw next-token LM. Use `mode: pretrain` with `text_field` instead.

---

## The cardinal rule

> Measure what you lose, not only what you gain.

Fine-tuning a post-trained model on one skill costs it others, and a loss curve shows none of it. Keep a small regression battery (instruction following, knowledge, code, plain chat) next to your target eval, and compare every data choice against a random subset of the same size.
