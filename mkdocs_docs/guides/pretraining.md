# Pretraining and Continued Pretraining

*Raw text with a loss on every token: continued pretraining (CPT) of a released model on a domain, a language or a body of knowledge, how to keep what the model already knew, and how to bring the result into an instruction-tuned model.*

---

!!! abstract "Use this guide when"
    You have a corpus of plain text (documents, books, code, web pages in your language) rather than conversations, and you want the model to learn what is in it. For conversations and agent traces read [Data Preparation](data.md) and [Agentic & Reasoning Models](agentic-training.md); for raw text mixed into SFT as a regulariser, `data.pretrain_replay_dataset` in the [configuration reference](../reference/configuration.md#data).

    Every key below is in the [configuration reference](../reference/configuration.md); every command in the [CLI reference](../reference/cli.md).

## In short

```bash
pgs prepare --config cpt.yaml                   # clean the corpus: duplicates, near-duplicates, benchmark leaks
pgs inspect --config cpt.yaml --num_samples 3   # every token of a raw-text sequence has a label
pgs fit --config cpt.yaml                       # the fastest memory setting that fits this GPU
pgs train --config cpt.yaml                     # train; re-run the same command to resume
```

```yaml title="cpt.yaml (the essentials)"
data:
  sources:
    - {dataset: data/corpus.jsonl, mode: pretrain, text_field: text}
  eval_dataset: data/heldout.jsonl
```

One epoch is one pass over every chunk of every document.

Raw text is a data **source** with `mode: pretrain`. The rest of the trainer (memory options, FSDP, checkpoints, resume, evaluation) is the one SFT uses. The full configs are [below](#a-minimal-config).

---

## Pretrain, continue pretraining, or fine-tune?

| You want | Use | Why |
|---|---|---|
| Knowledge, vocabulary or fluency that lives in text you cannot turn into conversations (a technical domain, a language, an internal corpus) | **Continued pretraining** of the **base** model, then SFT on top, then [merge into the instruction-tuned model](#continued-pretraining-of-an-instruction-tuned-model) | Raw text teaches what to say; it does not teach how to answer, follow instructions or call tools |
| A behaviour: answering, a format, tool calls, reasoning, refusals | **SFT** on conversations ([Data Preparation](data.md)), then RL or distillation | Loss only on the assistant's turns; the prompt is context, not a target |
| A new model from random weights | `model.init: random` ([Training from random weights](#training-from-random-weights)) | The architecture comes from a `config.json`, the weights from a seed. The trainer runs it like any other run, but none of the defaults are tuned for it and it has only been tested at toy scale |
| A large teacher's knowledge on your corpus | [Off-policy distillation](offpolicy-distillation.md) with `mode: pretrain` sources | Pretraining-mode sources distill every token against the teacher's distribution |

Continued pretraining usually trains on far more tokens than SFT, and its side effects are the ones a training loss hides best. Run it when the missing thing really is in raw text, and measure what it costs (see [Held-out loss and forgetting](#held-out-loss-and-forgetting)).

---

## How raw text is trained

A source with `mode: pretrain` reads one text column (`text_field`, default `text`) from each row:

- **Every token gets loss.** There is no chat template and no masking: the whole string is next-token prediction.
- **The tokenizer adds what it always adds.** Documents are tokenized with `add_special_tokens=True`, so a tokenizer that prepends a BOS token (Llama) prepends it to every document; one that adds nothing (Qwen) adds nothing.
- **Every document ends with an end-of-document token.** The tokenizer's EOS token is appended to each document (`append_eos`, on by default; a document whose text already ends with it keeps one). It is trained like any other token, so the model learns where documents end. See [The end-of-document token](#the-end-of-document-token) for chat models, whose EOS is the end-of-turn token.
- **Long documents are split, not truncated.** A document longer than `data.max_seq_length` becomes consecutive `max_seq_length`-token chunks; each chunk is its own training sequence: the second chunk of a document does not see the first. Only the document's last chunk carries the end-of-document token. A last chunk that would be shorter than 16 tokens (for a document of exactly two sequence lengths, the end-of-document token alone) reaches back into the previous chunk to 16 tokens, so the end of the document is always trained and at most 15 tokens are trained twice; with `append_eos: false` such a tail is dropped.
- **Packing keeps sequences apart.** With `data.packing: true`, whole chunks and short documents are packed into blocks of up to `max_seq_length` tokens, and `position_ids` restart at every document, so no document attends to another ([packing](../getting-started/first-training.md#packing-when-to-use-it)). The loss never asks the last token of a document to predict the first token of the next one. For raw text, where most documents are shorter than the sequence length, packing fills each sequence instead of padding short documents.
- **Only through `data.sources`.** The single `data.dataset` is always read as conversations. A list of one source is fine.

!!! warning "Don't feed a rendered conversation as `text`"
    A chat transcript in a `text` field trains on the whole prompt, user turns included, because raw text masks nothing. Conversations go in a `mode: sft` source with `messages`; both kinds can be mixed in one run ([What gets loss](data.md#what-gets-loss-sft-masking)).

Check before training what a sequence looks like:

```bash
pgs inspect --config cpt.yaml --num_samples 3     # tokens and labels, as the trainer receives them
pgs validate --config cpt.yaml --num_samples 200  # masking statistics
```

`pgs validate` (and the masking check of `pgs diagnose --mode pre`) warns that the samples "train on ALL tokens" and that the train ratio is very high. For raw text that is the expected result, not a masking bug; the exit code stays 0.

### The end-of-document token

```yaml
data:
  sources:
    - {dataset: data/corpus.jsonl, mode: pretrain}                               # the tokenizer's EOS (default)
    - {dataset: data/corpus.jsonl, mode: pretrain, append_eos: "<|endoftext|>"}  # a named token
    - {dataset: data/corpus.jsonl, mode: pretrain, append_eos: false}            # nothing appended
```

| `append_eos` | Appended to every document |
|---|---|
| `true` (default) | `tokenizer.eos_token`. A tokenizer without one appends nothing and logs a warning |
| a token string | That token; it must be exactly one token of the vocabulary, or the run stops with an error |
| `false` | Nothing: the behaviour before this option existed |

- **Chat models.** The EOS token of an instruction-tuned tokenizer is usually its end-of-turn token (`<|im_end|>` for Qwen chat models). Ending raw documents with it teaches that documents end like turns, and the trainer logs a warning when the default resolves to a token written in the chat template; name the base model's document separator instead (`append_eos: "<|endoftext|>"` for Qwen, `"<|end_of_text|>"` for Llama 3), and check with `pgs inspect` what the last token of a sequence is.
- **Packing.** In a packed block the token closes each document: `[doc A … EOS][doc B … EOS]`. Position ids restart at every document and attention stays inside it. EOS is predicted from its own document's last token, and no loss is placed on EOS predicting the first token of the next document.
- **Held-out sets** read as raw text (`data.eval_dataset` in a raw-text run, `eval_sources` with `mode: pretrain`) get the same token, so the evaluated loss includes predicting the end of each document. A held-out loss measured before this option existed is not comparable to one measured with it: compare against the baseline the run prints before its first step.
- **Replay text** (`data.pretrain_replay_dataset`, raw text mixed into SFT) gets no end-of-document token: SFT runs use chat tokenizers, and the text is trained as it always was. Use a `mode: pretrain` source with `append_eos` when you want one.
- A [pre-tokenized cache](data.md#pre-tokenized-cache) written without the token is rebuilt. Resuming a raw-text run from a checkpoint written by an earlier version logs a warning: the tokens and the epoch length have changed, so the position in the epoch is not the one the run stopped at (`append_eos: false` and `train.max_steps` keep the old stream's tokens and the run's length).

---

## Preparing the data

**Format.** A JSONL or parquet file with a text column, a directory of them on the Hub, or any Hugging Face dataset: `dataset` takes the same values as for chat data. Extra columns are ignored. With `data.streaming: true` (the default) local JSONL and parquet files are read as many shards in parallel; write parquet with several row groups so it can be split ([Run length and streaming](data.md#run-length-and-streaming)).

**Clean the corpus with `pgs prepare`.** For `mode: pretrain` sources it works on documents, without loading a model:

```yaml title="cpt.yaml"
data:
  sources:
    - {dataset: data/raw_corpus.jsonl, name: corpus, mode: pretrain, text_field: text}
preprocess:
  output_dir: ./prepared
  dedupe: true                 # exact duplicates (the default)
  near_dedupe: 0.8             # near-duplicates: Jaccard similarity of word 5-gram sets (0 = off)
  decontaminate:               # the evaluation sets you will report
    - benchmarks/domain_qa.jsonl
    - benchmarks/general.parquet
  decontaminate_ngram: 13
  decontaminate_fields: [question, answer]   # default: every string column
  min_chars: 200
  eval_holdout: 500            # whole documents, drawn after deduplication
```

```bash
pgs prepare --config cpt.yaml
# or without a config:
pgs prepare --data data/raw_corpus.jsonl --output prepared/corpus --mode pretrain \
    --near_dedupe 0.8 --decontaminate benchmarks/*.jsonl --min_chars 200 --eval_holdout 500
```

Each source is written to `prepared/<name>/`: `scored_data.parquet` (the kept documents, every column), `eval_data.parquet` (the held-out documents), `dropped.jsonl` (every removed row with its reason and, for a duplicate, the row that was kept) and `prepared_meta.json` with the counts. The command logs the `data.sources` entries to train on; a source can point at the directory (`dataset: prepared/corpus`) and the held-out file goes in `data.eval_dataset` or an `eval_sources` entry. In a sources file for `pgs prepare-multi`, `mode: pretrain` entries are cleaned the same way next to the chat sources (`--decontaminate`, `--near_dedupe`, `--min_chars`).

What each check does, in the order they run:

| Reason in the report | Removed | Option |
|---|---|---|
| `empty_text` | Rows without a string in `text_field`, or only whitespace | always |
| `too_short` | Documents with fewer than `min_chars` characters | `min_chars` |
| `contaminated` | Documents that share a run of `decontaminate_ngram` words with the `decontaminate` files: every string column and nested value, so questions and answers both, or only the columns in `decontaminate_fields`; case and punctuation ignored. When every column is read, an n-gram found in more than 20% of a file's rows (files of 10 rows or more) is not matched: it is a shared instruction, system prompt or few-shot prefix, and matching it would remove every document that quotes it (`decontaminate_shared_share`) | `decontaminate` |
| `duplicate` | Documents whose text is the same after Unicode normalisation, lower-casing and collapsing whitespace; the first is kept | `dedupe` |
| `near_duplicate` | Documents whose sets of word 5-grams have an estimated Jaccard similarity of at least `near_dedupe` with an earlier kept document (MinHash, 128 hashes, candidates by locality-sensitive hashing) | `near_dedupe` |

Read the report before training:

- **Name the columns that hold the items.** By default every string of an evaluation row is read, ids and metadata included. `decontaminate_fields: [question, answer]` reads only those columns (a file without any of them is an error), and every n-gram of them is matched: the shared-n-gram rule is off when the columns are named.
- **`decontaminate_shared_share`** sets the rule yourself: a share between 0 and 1 applies it (also with `decontaminate_fields`, e.g. when the question column repeats an instruction), `0` turns it off (match every n-gram, the strictest setting), `-1` is the default described above. `prepared_meta.json` reports how many shared n-grams were left out (`decontamination.shared_ngrams_ignored`).
- **`contaminated` counts are usually small and matter most.** A domain corpus scraped from the web often contains the test sets of that domain. Evaluation strings shorter than the n-gram give no n-gram and protect nothing: `prepared_meta.json` counts them (`decontamination.strings_shorter_than_ngram`); lower `decontaminate_ngram` for benchmarks of short items, at the price of more false matches on common phrases.
- **`near_dedupe` is an estimate.** With 128 hashes the similarity of a pair is known to about ±0.04 at 0.8, so pairs close to the threshold fall on either side. 0.8 removes mirrored pages and lightly edited copies; lower values start removing documents that share boilerplate (licence headers, templates), which may or may not be what you want: read a sample of `dropped.jsonl` first.
- **Duplicates are removed inside a source, not across sources**, and held-out documents are drawn after deduplication, so no held-out text is trained on under another copy from the same source.
- **Memory and time.** Everything is held in memory and runs in one process: one hash per evaluation n-gram, one signature per kept document. Sized for corpora of up to a few million documents; for a web-scale crawl use a dedicated tool and feed its output here.

Still yours to do:

- **Filter.** Other languages than the one you want and extraction debris (navigation menus, tables flattened into noise) cost tokens and teach nothing; `min_chars` only removes the shortest documents.
- **Document ends** are added at training time ([The end-of-document token](#the-end-of-document-token)): do not write the EOS string into the text.

The same checks in plain Python, as a starting point when you need to change them (exact duplicates, 13-gram decontamination, a held-out split of 500 documents):

```python title="clean_corpus.py"
import hashlib, json, random, re

def norm(text):
    return re.sub(r"\s+", " ", text.lower()).strip()

def ngrams(text, n=13):
    words = norm(text).split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}

# n-grams of every benchmark you will report (questions and answers)
bench = set()
for line in open("benchmarks.jsonl"):
    bench |= ngrams(json.loads(line)["text"])

seen, kept, dropped = set(), [], {"duplicate": 0, "contaminated": 0, "short": 0}
for line in open("raw_corpus.jsonl"):
    text = json.loads(line)["text"]
    if len(text) < 200:
        dropped["short"] += 1
        continue
    key = hashlib.sha1(norm(text).encode()).hexdigest()
    if key in seen:
        dropped["duplicate"] += 1
        continue
    seen.add(key)
    if ngrams(text) & bench:
        dropped["contaminated"] += 1
        continue
    kept.append(text)

random.Random(0).shuffle(kept)
heldout, train = kept[:500], kept[500:]
for path, docs in [("heldout.jsonl", heldout), ("corpus.jsonl", train)]:
    with open(path, "w") as f:
        for text in docs:
            f.write(json.dumps({"text": text}) + "\n")
print(len(train), "train,", len(heldout), "held out, dropped", dropped)
```

It holds every n-gram of the benchmarks and every document hash in memory: fine for millions of documents, not for a web-scale corpus.

---

## Mixing sources

A run can mix several raw-text sources, and raw text with conversations, in one `data.sources` list ([multi-source mixing](data.md#multi-source-mixing)):

```yaml
data:
  sources:
    - {dataset: data/domain.parquet, name: domain, mode: pretrain, text_field: text, weight: 0.8}
    - {dataset: data/general.parquet, name: general, mode: pretrain, text_field: text, weight: 0.2}
```

- **`weight` is by default a share of training sequences, not of tokens.** Each draw takes one sequence (a document, or one `max_seq_length` chunk of a long one) from a source chosen by weight. A source of long documents contributes about `max_seq_length` tokens per draw, one of short documents far fewer, so with sources of different lengths the token shares differ from the weights.
- **`data.mix_weights: tokens` makes the weights shares of the tokens.** Replay ratios in the literature, and budgets, are stated in tokens. With `mix_weights: tokens` the example above trains 80% of its tokens on the domain corpus whatever the lengths of the documents (see the table below). Every source needs a `weight`.
- **Without weights the mix is one shuffled concatenation.** Each source is drawn in proportion to the training sequences it holds (its rows, times the chunks its documents give), so every source gets about one pass per epoch.
- The startup log gives each source's share (`of sequences`, `of tokens` or `of rows`) and its rows.

| `data.mix_weights` | `weight: 0.8` means | A source is drawn with probability proportional to | Use when |
|---|---|---|---|
| `sequences` (default) | 80% of the training sequences | its weight | sequences are the unit you reason in (conversations in SFT), or the sources have similar lengths |
| `tokens` | 80% of the tokens trained on | its weight ÷ the mean length of its sequences | raw text of different lengths: replay shares, token budgets |

How `tokens` is computed: the mean sequence length of each source is measured on the sequences drawn from it so far in the epoch (per data-loading worker), and the draw probabilities follow it. Until a source has been drawn once it takes the mean of the others, so the first few draws of an epoch are slightly off and the shares converge within a few hundred sequences. Every token of a sequence counts (for a conversation source in the mix, the prompt too, not only the trained tokens). The size estimate that sets the LR schedule uses the same rule.
- **Replay general text.** A share of text like the model's original pretraining data (a general web or encyclopedic corpus in the languages the model already serves) slows the loss of general ability. Ibrahim et al. ([arXiv:2403.08763](https://arxiv.org/abs/2403.08763)) report that 5% replay was enough for a mild distribution shift and that a stronger one (a new language) needed about 25%. These are their numbers, not measured here: measure your own trade-off with a general held-out set (next sections).
- **Conversations in the mix** (`mode: sft` sources next to raw text) keep the chat behaviour of a model that has one. See [Continued pretraining of an instruction-tuned model](#continued-pretraining-of-an-instruction-tuned-model) for why that is usually not enough.

!!! note "What an epoch is"
    With the default `data.mix_epoch: total`, an epoch is as many **rows** as the sources hold together, and a document counts when its last chunk has been drawn. A document split into several chunks therefore counts once, with all its chunks:

    - **One raw-text source:** an epoch is one pass over every chunk of every document (10 documents of 3 chunks each: 30 sequences). `mix_epoch: first_exhausted` gives the same epoch.
    - **A mix without weights:** about one pass over every source.
    - **A mix with weights:** the same number of rows, distributed by the weights: a source weighted above its share is repeated in a new order, one below it contributes a random subset that changes every epoch. To fix the budget in tokens instead, set `train.max_steps`, with `train.epochs` large enough that the step limit ends the run.

    Earlier versions counted one sequence per row, so an epoch over a corpus of long documents read only part of it (a third when documents gave three chunks each). A run sized in epochs on long documents is now longer by that factor; a run with `train.max_steps` is unchanged.

---

## A minimal config

One GPU, one corpus, a held-out set of documents:

```yaml title="cpt.yaml"
model:
  name_or_path: Qwen/Qwen3-0.6B-Base   # the base model, not the instruct one

data:
  sources:
    - dataset: data/corpus.jsonl
      name: corpus
      mode: pretrain
      text_field: text
  max_seq_length: 4096
  packing: true
  streaming: true
  eval_dataset: data/heldout.jsonl   # read as raw text: every source is mode: pretrain
  eval_samples: 200                  # sequences (chunks), not documents
  eval_every: 200

train:
  output_dir: ./runs/cpt
  epochs: 1
  per_device_batch_size: 4
  gradient_accumulation_steps: 16    # 4 x 16 x 4096 = 262k tokens per step (at full packing)
  learning_rate: 2.0e-5
  min_learning_rate: 2.0e-6
  warmup_ratio: 0.01
  lr_scheduler: cosine
  save_steps: 500
  resume_from: auto

logging:
  use_wandb: false
  use_trackio: false
```

One epoch is one pass over every chunk of every document. `data.eval_dataset` is read as raw text (all-token loss, `text_field` and `append_eos` of the first source) only when **every** training source is `mode: pretrain`. In a run that also has conversation sources it is read as conversations; use `data.eval_sources` with `mode: pretrain` instead (next section).

Before the first run:

```bash
pgs diagnose --config cpt.yaml --mode pre --json   # memory estimate, masking, config sanity
pgs fit --config cpt.yaml                          # measured: checkpointing mode and SeCO chunk that fit
```

---

## A scaled config

Eight GPUs, a domain corpus with general replay, per-source held-out losses, a budget fixed in steps:

```yaml title="cpt_8b.yaml"
model:
  name_or_path: meta-llama/Llama-3.1-8B
  attn_implementation: flash_attention_2   # packed sequences need no explicit mask
  compile: true

data:
  sources:
    - {dataset: data/domain.parquet, name: domain, mode: pretrain, text_field: text, weight: 0.8}
    - {dataset: data/general.parquet, name: general, mode: pretrain, text_field: text, weight: 0.2}
  mix_weights: tokens   # 80% / 20% of the tokens, whatever the documents' lengths
  max_seq_length: 8192
  packing: true
  streaming: true
  num_workers: 8
  eval_every: 250
  eval_sources:
    - {name: domain, dataset: data/domain_heldout.parquet, mode: pretrain, text_field: text, weight: 0.5, samples: 200}
    - {name: general, dataset: data/general_heldout.parquet, mode: pretrain, text_field: text, weight: 0.5, samples: 200}

train:
  output_dir: ./runs/cpt_8b
  max_steps: 20000      # 2 x 8 x 8192 x 8 GPUs = 1.05M tokens per step: about 21B tokens
  epochs: 100           # large enough that max_steps ends the run
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-5
  min_learning_rate: 3.0e-6
  warmup_ratio: 0.01
  lr_scheduler: wsd
  save_steps: 1000
  keep_checkpoints: 3
  resume_from: auto

parallel:
  fsdp: true

memory:
  chunked_loss: true
  loss_num_chunks: 8
```

```bash
torchrun --standalone --nproc_per_node=8 -m palingenesis.train --config cpt_8b.yaml
```

Launching on several GPUs and nodes is in [Multi-GPU](multi-gpu.md#launch) and [Multi-Node & SLURM](multi-node.md).

**Pre-tokenized cache.** `data.pretokenize: true` tokenizes, chunks, mixes and packs the stream once and reads tensors on later runs ([pre-tokenized cache](data.md#pre-tokenized-cache)). It stores **one epoch** of the stream and reshuffles that same set every later epoch, so use it when an epoch is the whole budget (one source, or a mix without weights), not with the `max_steps` plus many epochs pattern above, where every epoch would repeat the cached subset.

---

## Learning rate, warmup, schedule and token budget

**Tokens per optimizer step** = `per_device_batch_size` × `max_seq_length` × `gradient_accumulation_steps` × data-parallel GPUs (with context parallelism, GPUs ÷ `cp_degree`), times the packing fill rate. The budget in tokens divided by that is `train.max_steps`. `train/tokens_total` in the log counts the trained tokens, so compare runs by tokens, not steps.

**How many tokens.** For training from scratch the usual reference is Chinchilla's compute-optimal ratio of about 20 tokens per parameter (Hoffmann et al., [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)); released models are trained far beyond it. For continued pretraining the budget is set by the corpus: one pass over clean, deduplicated text is the default. Muennighoff et al. ([arXiv:2305.16264](https://arxiv.org/abs/2305.16264)) report that up to about four epochs of repeated data are nearly as useful as fresh data and that returns fall quickly after that.

**Peak learning rate.** There is no measured default for raw text in this library: the `train.learning_rate` default (`2e-5`) is a fine-tuning value. Continued pretraining moves the model more than SFT does, and the peak LR is the main knob between learning the corpus and forgetting the rest. Gupta et al. ([arXiv:2308.04014](https://arxiv.org/abs/2308.04014)) report that a higher peak adapts more to the new data and forgets more of the old. To choose it:

1. Start in the fine-tuning range (`1e-5` to `5e-5` for full fine-tuning of 1B–8B models; the range `pgs diagnose` checks against) and run short pilots of a few hundred steps at ×3 spacing.
2. Keep the highest LR whose **general** held-out loss stays within what you accept while the domain loss falls. A lower LR for longer is the safer side.
3. `pgs diagnose` flags a learning rate above `1e-3` as "very high for SFT": that check is tuned for fine-tuning, but at continued-pretraining scale such a rate is far above any of the starting points here, so treat the warning as meant.

**Warmup.** A released model ends its pretraining at a low learning rate; raising it again ("re-warming") makes the loss on the original distribution rise at first, a known effect of continued pretraining (Gupta et al., above). Keep a linear warmup (`train.warmup_ratio`; `0.01` to `0.05`): `pgs diagnose` warns when it is 0.

**Schedule.** `train.lr_scheduler` (see [schedulers](../reference/optimizers.md#schedulers)):

| Schedule | Shape | When |
|---|---|---|
| `cosine` (default) | warmup, cosine decay to `min_learning_rate` | the baseline; any run whose length is fixed in advance |
| `wsd` | warmup, peak LR for 80% of the post-warmup steps, then a power decay over the last 20% | long runs with a fixed `max_steps`; the decay is always the last 20% of the horizon |
| `power_decay` | warmup, then $(1 - z/N)^4$ decay | derived for pretraining in a scaling-law model ([arXiv:2602.06797](https://arxiv.org/abs/2602.06797)); not benchmarked here |
| `constant` | warmup, then flat | pilots and LR sweeps only: the model never anneals |

The decaying schedules end at `min_learning_rate`; with the defaults the floor is a tenth of the peak.

**Optimizer.** `adamw` is the default and the one to compare against. Muon (`train.optimizer: muon`) is reported faster than AdamW in pretraining and is RMS-matched to AdamW, so the same `learning_rate` and `weight_decay` apply; on a model that was pretrained with AdamW, compare it against AdamW on a pilot before a long run ([Muon](../reference/optimizers.md#muon)). LoRA is a poor fit: a corpus carries more than an adapter can hold ([LoRA guide](lora.md#when-lora-wins-and-when-full-fine-tuning-is-better)).

---

## Held-out loss and forgetting

The training loss of a continued pretraining run falls almost no matter what; the question is what the model loses on the way. Measure three things with `data.eval_sources`, each scored on its own:

```yaml
data:
  eval_every: 250
  eval_sources:
    - {name: domain, dataset: data/domain_heldout.jsonl, mode: pretrain, text_field: text, weight: 0.5, samples: 200}
    - {name: general, dataset: data/general_heldout.jsonl, mode: pretrain, text_field: text, weight: 0.5, samples: 200,
       regression_floor: 2.4}
    - {name: chat, dataset: data/chat_heldout.jsonl, mode: sft, messages_field: messages, weight: 0.0, samples: 100}
```

- **Domain** (`mode: pretrain`): held-out documents of the corpus. It should fall; if it stops falling the run is done.
- **General** (`mode: pretrain`): text like the model's original data, in the languages it serves. It rises at the start of the run (re-warming) and should settle. A steady rise is forgetting: lower the LR, add replay, or stop earlier.
- **Chat** (`mode: sft`, only if the model has one): assistant-only loss on conversations. Weight 0 keeps it out of the composite that picks `best/`, while its loss is still logged.

!!! note "`eval_sources` files"
    An `eval_sources` entry is a local `.jsonl`, `.json` or `.parquet` file, a `pgs prepare` output directory, or a Hub dataset (its `split`, default `test`). A source that cannot be read, has an unknown `mode` or gives no usable sample (a wrong `text_field`, say) stops the run with a config error that names it: a configured evaluation set is never skipped.

What you get:

- `eval/<name>/loss` and `eval/<name>/ppl` per source, the weighted composite `eval/loss`, and on the console line `eval_sources=domain:<loss>,general:<loss>,chat:<loss>`.
- The **baseline before the first step**, printed the same way and marked `(before training)`: compare every later value with it, not with the previous evaluation.
- `regression_floor`: an absolute loss above which a source counts as regressed (a warning names it; `eval/regressions` counts them). Set it a margin above that source's baseline.
- `best/` holds the checkpoint with the lowest composite `eval/loss`; choose the weights so the composite reflects what you would adopt.
- `health/weight_drift_max`, the largest relative change of a weight from its initial value, every 100 steps ([weight drift](../reference/health.md#weight-drift-healthweight_drift_max)).

A loss is not a capability. Before adopting a checkpoint, evaluate it on downstream benchmarks of both the domain and the skills you want to keep (with your evaluation harness: palingenesis measures losses, not benchmark scores), and compare against the model you started from.

Follow a run from its console log:

```bash
pgs train --config cpt.yaml 2>&1 | tee train.log
pgs monitor --log_file train.log --brief      # one line: status, loss trend, throughput, time left
pgs loss --log_file train.log                 # the whole curve
```

---

## Resuming

Long runs get interrupted. With `train.resume_from: auto`, running the same command again resumes from the newest complete `step-N/` checkpoint in `train.output_dir`, or starts fresh if there is none:

- Model, optimizer, LR schedule and position in the data are restored, and the metrics continue in the same tracker run. Pass an explicit `step-N` path to `resume_from` to resume from another checkpoint.
- **Mid-epoch, the data before the saved position is read again and skipped** (not trained), so the restart of a long epoch takes time proportional to how far into it the run was. A [pre-tokenized cache](data.md#pre-tokenized-cache) makes that a read of tensors.
- **Disk:** a step checkpoint holds the weights and the optimizer state (about 3× the fp32 weights with AdamW). `train.keep_checkpoints` (default 5) of them are kept; `best/` and `final/` are never deleted. Size `save_steps` by how much compute you can afford to lose.
- Several GPUs or nodes: a sharded checkpoint can resume on another number of ranks (2 or more), with caveats on the data order and the effective batch ([checkpointing across nodes](multi-node.md#checkpointing-across-nodes)); `output_dir` must be on a filesystem every node sees.

---

## Long context and several GPUs

Continued pretraining is often also where a model learns longer sequences. Memory for activations grows with `max_seq_length`; the options, in the order to try them:

| Situation | Setting | Guide |
|---|---|---|
| Find the fastest setting that fits | `pgs fit --config cpt.yaml` (checkpointing modes and SeCO chunk sizes, measured on rows of full length) | [Single GPU](single-gpu.md#trading-speed-for-memory-pgs-fit) |
| One GPU, sequences beyond what checkpointing fits | `memory.seco: true` (exact gradients, chunk by chunk; works with packing) | [Long Sequences (SeCO)](long-sequences.md) |
| Several GPUs, model or optimizer state too large for one | `parallel.fsdp: true` | [Multi-GPU](multi-gpu.md) |
| Several GPUs, one sequence too long for one GPU | `parallel.context_parallel: true` with `fsdp: true`, `data.packing: false` (refused with packing) | [Context Parallel](multi-gpu.md#context-parallel) |
| Large vocabulary | `memory.chunked_loss: true` (the default), more `memory.loss_num_chunks` | [Configuration](../reference/configuration.md#memory) |

Two things specific to raw text: with packing on Qwen3.5 and other linear-attention hybrids, install the `hybrid` extra or the trainer refuses to pack ([Installation](../getting-started/install.md)); and with context parallelism, where packing is off, every document shorter than `max_seq_length` is one padded sequence, so set `max_seq_length` to the length you train for, not far above it.

---

## Continued pretraining of an instruction-tuned model

The model you deploy is usually an instruction-tuned or agent checkpoint, and it is tempting to run continued pretraining on it directly. Don't:

- Raw text has no turns, no tool calls and no end-of-turn tokens, and continued pretraining overwrites exactly the behaviour post-training added. In our tests on a small model, continued pretraining on an agent checkpoint **erased its tool calling** and cut its grade-school math accuracy by about a third.
- Mixing some conversations into the run slows this down but does not reproduce the post-training (RL, preference tuning) that produced the behaviour.

Train the knowledge as an **expert from the base model** and add it to the instruction-tuned model by [task arithmetic](merging.md#operations):

1. **Continued pretraining of the base** (the checkpoint the instruction-tuned model was post-trained from), as above.
2. **A short SFT of that checkpoint** on conversations, in the domain or language of the corpus where possible. A delta shaped by SFT combined with the instruction-tuned model in our tests; a delta from continued pretraining alone did not (below).
3. **Add the expert's change** to the instruction-tuned model, at a small strength:

    $$
    W = W_{\text{instruct}} + \lambda\,(W_{\text{expert}} - W_{\text{base}})
    $$

    ```bash
    pgs merge task-arithmetic --base BASE INSTRUCT:1 runs/expert/final:0.3 -o runs/instruct_domain_0.3
    ```

    The output keeps the instruction-tuned model's tokenizer, chat template and configs (the first model is the template).
4. **Sweep λ and gate.** Merges take seconds. Evaluate a few values (for example 0.2, 0.3, 0.4) on the domain evaluation **and** on the instruction-tuned model's own battery (instruction following, tool calls, math, chat), and adopt the largest λ that keeps every gate ([Choosing lam](merging.md#choosing-lam-and-adopting-a-merge)). λ between 0.25 and 0.4 worked in our tests.
5. **Train on from the merge** if needed (SFT, RL or distillation), with the merge as the starting point and, in RL, as the KL reference, so later stages keep what it added.

Rules that held at small scale (re-check them at yours):

- **A delta from continued pretraining alone, added at full strength (λ = 1), collapsed instruction following** and made the model loop. Published "chat vector" transfers that work at 70B did not transfer to a sub-1B model.
- **Small λ, SFT-shaped expert, from the same base.** Base, expert and instruction-tuned model must share the architecture, the tokenizer and the origin.
- **A merge cannot remove a regression the expert already has**: evaluate the expert on its own first.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| An epoch is several times longer than in an earlier version | An epoch now covers every chunk of every document; it used to count one sequence per row | Expected ([What an epoch is](#mixing-sources)); fix the budget with `train.max_steps` |
| A weighted source takes far more (or fewer) tokens than its weight | Weights are shares of sequences by default | `data.mix_weights: tokens` |
| `Data source '…' yielded 0 usable examples` | Wrong `text_field`, an empty file, or a source without `mode: pretrain` (read as conversations) | Check the column name with `pgs inspect`; set `mode: pretrain` |
| Raw-text eval set gives no `eval/loss` in a mixed run | `data.eval_dataset` is read as conversations unless every training source is `mode: pretrain` | `data.eval_sources` with `mode: pretrain` |
| General held-out loss jumps in the first steps | Re-warming the learning rate | Expected if it settles; if not, lower `learning_rate` or lengthen `warmup_ratio` |
| General held-out loss keeps rising | LR too high, too little replay, or too many tokens | Lower LR, raise the replay weight, adopt an earlier checkpoint (`best/` with a general-weighted composite) |
| Chat model lost tool calls or instruction following | Continued pretraining on the instruction-tuned checkpoint | CPT the base, SFT it, merge at small λ ([above](#continued-pretraining-of-an-instruction-tuned-model)) |
| Generations never end | `append_eos: false`, or a base model that was never fine-tuned to stop | Keep `append_eos` on ([above](#the-end-of-document-token)); SFT for chat use |
| A chat model ends its turns badly after raw-text training | Documents were closed with the tokenizer's EOS, which is the end-of-turn token | `append_eos: "<|endoftext|>"` (the base model's document separator) |
| `append_eos: … is not a token of this tokenizer's vocabulary` | The named string is not exactly one token | Use a token of the tokenizer (`tokenizer.eos_token`, a special token) |
| `pgs validate` warns "train on ALL tokens" | Expected for raw text | Nothing to fix |
| `pgs diagnose` warns that the LR is "very high for SFT" | LR above `1e-3` | Far above the continued-pretraining starting points: lower it unless you are deliberately [training from random weights](#training-from-random-weights) |
| OOM at the target `max_seq_length` | Activations grow with the sequence | `pgs fit`; `memory.seco` on one GPU, context parallelism on several |
| Packing refused on a hybrid model | Linear-attention packing kernels missing | Install the `hybrid` extra |
| Context parallelism refused | Packing is on | `data.packing: false` with context parallelism |
| Restart after a crash takes long before the first step | The resumed epoch reads and skips the data already trained on | Expected; `data.pretokenize: true` makes it fast when one epoch is the whole budget |
| Every epoch trains on the same subset | Pre-tokenized cache with several epochs of a mixed stream | Turn `pretokenize` off for `max_steps`-bounded mixes |
| `data.eval_sources: source '…' …` stops the run | The file is missing or of another format, the `mode` is unknown, or no row gives a sample (wrong `text_field` / `messages_field`) | Fix the entry: the message names the source and the cause |
| `model.init: random` warns about the learning rate | `train.learning_rate` at or below `1e-4`, a fine-tuning value | Raise it for training from scratch ([below](#training-from-random-weights)) |
| `pgs prepare` removes almost nothing as `contaminated` although the benchmark is in the corpus | Benchmark items shorter than the n-gram (`strings_shorter_than_ngram` in `prepared_meta.json`) | Lower `preprocess.decontaminate_ngram` |

---

## Training from random weights

`model.init: random` builds the architecture described by `model.name_or_path`'s `config.json` and initialises its weights from a seed, instead of loading a checkpoint:

```yaml title="scratch.yaml"
model:
  name_or_path: ./arch     # config.json + the tokenizer files; no weights needed
  init: random
  init_seed: 0             # -1 (default): train.seed
  torch_dtype: float32

data:
  sources:
    - {dataset: data/corpus.parquet, mode: pretrain, text_field: text}
  max_seq_length: 2048
  packing: true

train:
  learning_rate: 6.0e-4
  min_learning_rate: 6.0e-5
  warmup_ratio: 0.02
  weight_decay: 0.1
  max_steps: 100000
  epochs: 1000
```

Make the directory from any architecture `transformers` knows, with a tokenizer whose size fits the config:

```python
from transformers import AutoTokenizer, LlamaConfig

tokenizer = AutoTokenizer.from_pretrained("my-tokenizer")
config = LlamaConfig(
    vocab_size=len(tokenizer), hidden_size=768, intermediate_size=3072,
    num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=2048,
    bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
)
config.save_pretrained("arch")
tokenizer.save_pretrained("arch")
```

`name_or_path` can also be a released model, to train its architecture from scratch: its weights are ignored.

- **Initialisation.** The weights are drawn by the architecture's own initialiser in `transformers` (for most decoders a normal distribution with the config's `initializer_range`), directly in `model.torch_dtype`, as a loaded checkpoint would be (parameters in that dtype, rotary-frequency buffers in float32). Every rank draws the same weights from `init_seed`, and the run's own seeding on CPU and GPU (data order, dropout) is unchanged by it.
- **Host memory.** Every rank builds the model on the CPU before it moves to its GPU, as when a checkpoint is loaded: one copy in `model.torch_dtype` per rank (an 8B model in bf16: about 16 GB of host RAM per rank). Use `bfloat16` weights, or fewer ranks per node, when host memory is short.
- **Checked at startup:** a tokenizer with more tokens than the config's `vocab_size` is refused. A multimodal checkpoint's config gives its language model alone.
- **Rejected combinations:** `dpo.enabled`, `multimodal.enabled`, `train.base_merge`, `train.freeze_non_attention`, `plugins.pre_rl` and scoring data with the model in `preprocess`: each of them needs trained weights to start from.
- **Resume and export** work as in any run: `resume_from: auto` continues from the last checkpoint, and `final/` loads with `from_pretrained`.
- **The defaults are fine-tuning defaults.** A learning rate at or below `1e-4` logs a warning: training from scratch usually takes a peak between `1e-4` and `1e-3` with AdamW, a real warmup, and a token budget far beyond anything in this guide ([how many tokens](#learning-rate-warmup-schedule-and-token-budget)). `pgs diagnose` checks its learning-rate range for fine-tuning and will flag such a rate.
- **Scope.** `pgs train` only (`pgs rl` and `pgs distill` start from a trained model). Tested here on toy models on CPU; nothing in the library is tuned or benchmarked for pretraining at scale, and there is no tokenizer training.

---

## Not in the library

What a pretraining user might expect and palingenesis does not provide:

- **A pretraining recipe.** `model.init: random` gives the mechanism ([above](#training-from-random-weights)); schedules, initialisation variants, tokenizer training and data pipelines for web-scale corpora are not provided or tuned.
- **Web-scale cleaning and quality filtering.** `pgs prepare` deduplicates and decontaminates corpora that fit in memory ([Preparing the data](#preparing-the-data)); language identification, quality classifiers and distributed deduplication are left to dedicated tools.
- **Cross-document context.** Packed documents never attend to each other, and the chunks of a long document are trained independently: there is no option to train on one concatenated stream.
