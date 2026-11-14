# Mixture-of-Experts Models

A Mixture-of-Experts (MoE) layer sends every token to a few of its experts, chosen by a small router. Training such a
model differs from training a dense one in three places: the routers need a balancing signal, RL has to route the
trained tokens the way the rollout engine did, and the expert weights (most of the parameters) change what is worth
communicating between GPUs. This page covers what the trainers do about each, for `pgs train` (SFT, DPO, off-policy
distillation), `pgs distill` and `pgs rl`.

!!! info "When to read it"
    You are training Qwen3-MoE, Qwen3.5-MoE, Qwen3-Next, Mixtral, GPT-OSS, a DeepSeek-V3-style model (DeepSeek, GLM,
    MiniMax, ...) or any other transformers model whose experts are stored as stacked weights. Dense models are not
    affected by anything here.

Nothing has to be configured: a MoE model is recognised when it loads, and its routers are balanced with the
coefficient of its own config. The options, with their defaults:

```yaml
moe:
  balance: true              # false: the routers train on the task loss alone, no statistics
  balance_loss_coef: -1      # -1: the model config's router_aux_loss_coef (at most 0.01); 0: off
  z_loss_coef: 0.0           # router z-loss; ST-MoE used 0.001
  bias_update_rate: -1       # -1: 0.001 for routers with a selection bias (DeepSeek-V3 family), else off
  log_layers: false          # also log moe/layer_<i>/... per layer
  experts_implementation: auto

parallel:
  expert_parallel: false     # multi-GPU: keep each rank's experts in place, exchange tokens (experimental)
```

The same `moe:` section exists in RL and distillation configs; RL adds `rollout.routing_replay`.

## Why the routers need a balancing signal

The task loss gives a router no reason to use all of its experts. An expert that receives few tokens gets few
gradients, improves less, and is chosen even less: routers can collapse onto a few experts, on long fine-tunes and in
RL above all (the fine-tuning sections of ST-MoE, [arXiv:2202.08906](https://arxiv.org/abs/2202.08906), report that
the balancing losses help fine-tuning too). Pre-training uses an auxiliary loss for this, and transformers models
carry its coefficient (`router_aux_loss_coef`). They add the loss only inside their own forward with
`output_router_logits=True` and labels. The trainers here compute their losses from the hidden states themselves
(chunked cross-entropy, SeCO, distillation, policy gradients), so before this page's `RouterBalancer` existed that
loss was silently absent.

## What is added

**Load-balancing loss.** For each MoE layer with $E$ experts,

$$
\mathcal{L}_\text{balance} = E \sum_{i=1}^{E} f_i \, P_i
$$

where $f_i$ is the number of tokens routed to expert $i$ divided by the number of tokens (not differentiable; it sums
to `top_k`), and $P_i$ is the mean router probability of expert $i$ over the tokens (differentiable). The loss is
averaged over the MoE layers and multiplied by `balance_loss_coef`.
{: .pg-where }

This is the Switch Transformer loss ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961), eq. 4 to 6). Its minimum,
at uniform routing, is `top_k`. For sigmoid routers (DeepSeek-V3) $P$ uses the scores normalised over the experts, as
in that model's own balance loss ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437)).

**The load is measured over the global batch.** $f$ counts the tokens of every data-parallel rank and of every
micro-batch of the optimizer step so far, not of one micro-batch on one GPU. Qwen's study of this choice
([arXiv:2501.11873](https://arxiv.org/abs/2501.11873)) found that balancing within small micro-batches forces every
micro-batch to use all experts evenly, which works against specialisation: going from a balance batch of 4 to 512
sequences improved their pre-training perplexity, and a count buffer over accumulation steps recovered the same gain
on few GPUs. The implementation follows their algorithm 1: only the count vector is communicated (one all-reduce per
micro-batch for all layers), each rank keeps its local $P$, and the counts accumulate over the gradient-accumulation
steps and restart after the optimizer step. With one micro-batch per step on one GPU this is the plain Switch loss.

**Router z-loss** (`z_loss_coef`), from ST-MoE: the mean over tokens and layers of
$\big(\log \sum_i e^{x_i}\big)^2$ for router logits $x$. It keeps the logits small, which that paper found to
stabilise training. Off by default, since released checkpoints were not trained with it.

**Selection-bias update** (`bias_update_rate`), the auxiliary-loss-free balancing of
[arXiv:2408.15664](https://arxiv.org/abs/2408.15664), used by DeepSeek-V3: after every optimizer step each expert's
selection bias moves by the rate, down if the expert received more than the mean number of tokens in that step, up if
fewer. The bias is added to the scores for the top-k choice only; the gate values use the unbiased scores. It is
applied to routers that already have this bias in their checkpoint (`e_score_correction_bias`: DeepSeek-V3 and the
models built on it), where it is on by default with DeepSeek-V3's rate, 0.001. For other families it is refused: a
bias kept outside the checkpoint would not reach vLLM or any other inference stack, and the model would route
differently there. The update accumulates in fp32 (a bf16 buffer would round the steps away) and uses the same
global counts as the loss.

### Defaults

| Model | Balancing loss | Selection bias |
|-------|---------------|----------------|
| Qwen3-MoE, Qwen3.5-MoE, Qwen3-Next, Mixtral, GPT-OSS (softmax routers) | on, coefficient `router_aux_loss_coef` of the model config (0.001 in these families) | not available |
| DeepSeek-V3 family (sigmoid routers with a selection bias) | off (no coefficient in the config; set `balance_loss_coef`, DeepSeek-V3 used 0.0001) | on, rate 0.001 |

With LoRA the selection bias belongs to the frozen base model: set `bias_update_rate: 0` unless you export merged
weights, or the adapter alone will not reproduce the routing it was trained with.

A model config's coefficient above 0.01 (the largest published value, Switch Transformer's) is treated as a
placeholder and capped, with a warning; set `balance_loss_coef` explicitly to use more. `moe.balance: false` removes
the hooks altogether.

!!! note "Difference from transformers' own auxiliary loss"
    transformers concatenates the router logits of all layers and computes one loss over them, so $f$ and $P$ are
    pooled over the layers and a layer can be unbalanced as long as another one compensates. Here every layer is
    balanced on its own and the losses are averaged. For a model with one MoE layer the two values are equal.

## How the loss reaches the routers

The trainers' forwards differ: one pass, activation checkpointing (each layer runs twice), SeCO (chunk by chunk, with
its own backward), a loss computed outside the model from hidden states. There is no single scalar to add a term to.
The balancer therefore injects the gradient of the two losses where the router logits are produced: a forward hook
keeps each layer's logits, and the gate values pass through an identity whose backward returns
$\partial \mathcal{L} / \partial\,\text{logits}$ for them. Since $f$ is a constant of the loss and both losses are sums
over tokens, this is the exact gradient of the formulas above. The tests compare it with the losses written out and
added to the task loss as one scalar, for every family and every path below.

| Path | Status |
|------|--------|
| Chunked loss, cut cross-entropy, plain forward | exact (tested) |
| Activation checkpointing, full and selective | exact: tokens are counted once, in the first pass (tested) |
| SeCO | exact: the load is that of the whole micro-batch, measured in SeCO's no-grad pass (tested). SpaCO scales the visited chunks' router gradients like their loss |
| Gradient accumulation | the load covers the step's micro-batches so far (tested) |
| Packing, right padding | padding tokens are left out of the loads, the losses and the statistics (tested) |
| FSDP, several ranks | one all-reduce of the counts per micro-batch; gradients equal one process on the whole batch (tested on 2 and 3 CPU ranks) |
| Context parallel | each rank counts its slice of the rows (not tested on GPUs) |
| `model.compile` | the hooks and the `grouped` / `padded` experts stay Python (graph breaks); gradients tested equal on CPU with the eager and `grouped` experts. transformers' `grouped_mm` compiles only with bf16 weights |
| RL prompt-sharing trees | the load is the shared prompt's; the losses are averaged over the group's tokens |

Cost: per MoE layer and micro-batch, a softmax over [tokens, experts] for the statistics and one more for the
gradient. See [What is measured](#what-is-measured-and-what-is-not).

## Metrics

Logged with every logging step, over the last optimizer step's tokens on all ranks. The log line shows the first four
as `moe[load_max=... cv=... dead=... ent=...]`.

| Metric | Meaning |
|--------|---------|
| `moe/load_max` | The busiest expert's load relative to uniform, in the worst layer. 1 is perfectly balanced; `experts / top_k` means one expert gets every token |
| `moe/load_cv` | Coefficient of variation of the expert loads, mean over layers |
| `moe/dead_experts` | Share of experts that received no token in the step, mean over layers |
| `moe/router_entropy` | Mean entropy of the router distribution divided by $\log E$, mean over layers. Near 0: the routers are certain, and collapsed if `load_max` is high too |
| `moe/balance_loss`, `moe/z_loss` | The two losses before their coefficients (`balance_loss` is `top_k` at uniform routing) |
| `moe/route_mismatch` | RL with routing replay: share of the engine's expert choices the trainer's own routers would not have made |
| `moe/rank_load_max` | Expert parallelism: the busiest rank's share of the routed tokens relative to an even split, worst layer |
| `moe/layer_<i>/...` | With `log_layers: true`: the first four per MoE layer |

Rising `load_max` with falling `router_entropy` is collapse: raise `balance_loss_coef` (0.01 is Switch
Transformer's value), add `z_loss_coef: 0.001`, or lower the learning rate.

## RL: routing replay

In RL the rollout engine samples a token and reports its log-probability; the trainer recomputes the log-probability
of the same token and uses the ratio. For a MoE policy the two do not even run the same network: the engine and the
trainer pick different experts for part of the tokens, because their kernels and precision differ and, with
`rollout.max_staleness > 0`, because the weights do. Measured on Qwen3-30B-A3B by others: about 10% of the routers
choose differently and 94% of the tokens differ in at least one layer between SGLang and Megatron
([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)); about 10% of the activated experts change after one
gradient update ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071), section 5.3). That paper's remedy, Rollout
Routing Replay (R3), is implemented here:

1. The engine returns the experts it routed every token to. vLLM does this with `enable_return_routed_experts`
   (`[tokens - 1, MoE layers, top_k]`, kept with its prefix cache); `rollout.backend: vllm` and `vllm_server` switch
   it on, and `hf` records it with hooks.
2. The trajectory keeps the record of every position, turn after turn. Positions a later turn prefills again keep
   the record they were sampled under.
3. The trainer routes each token to the recorded experts and computes their gate values from its own router logits
   (for softmax routers: the softmax restricted to the recorded experts, R3's formula). The router therefore still
   receives gradient, through the gate values.

```yaml
rollout:
  routing_replay: auto    # on for MoE policies; "off"; "on" raises when the policy has no MoE layers
  prefix_caching: true    # the vLLM backends need it to return routing
```

`moe/route_mismatch` reports how often the trainer's own routers would have chosen differently, per step. On the
first step of an on-policy run it measures the engine against the trainer on identical weights. The token-level
safeguards of the objective stay in place (`policy/mismatch_k3`, the ratio caps, `loss.seq_mask`): replay removes the
routing part of the mismatch, not the rest.

Limits:

- With replay on, prompt-sharing trees (`train.prompt_sharing`) are off: the record is per rollout.
- A position without a record (an engine that returned none, a turn generated before a restart) is routed by the
  trainer itself.
- The record costs host memory: positions x MoE layers x top_k bytes per trajectory (one byte per choice up to
  256 experts), and every turn's request returns the routing of its whole context. For a 48-layer top-8 model that
  is 384 bytes per token.
- With `routing_replay: auto` a colocated vLLM that refuses the option is restarted without it (a warning is
  logged, and no `moe/route_mismatch` appears); `on` makes that an error. A separately launched server
  (`rollout.url`) must be started with `--enable-return-routed-experts` itself.
- vLLM refuses `enable_return_routed_experts` with pipeline parallelism and KV connectors.
- GSPO's sequence-level ratio ([arXiv:2507.18071](https://arxiv.org/abs/2507.18071)) was designed to make MoE RL
  stable without replay; `loss.type: gspo` and replay can be combined.

On-policy distillation needs no replay: its loss scores the student's current distribution on its own samples and
has no importance ratio against the engine. The balancing above applies to it.

## Expert compute

transformers keeps a layer's experts in stacked parameters and, since 5.x, runs them by default as a grouped GEMM over
tokens sorted by expert (`grouped_mm`), so `pgs train` was already using it. `moe.experts_implementation` selects
another forward over the same parameters; checkpoints keep transformers' layout in every case.

| Value | What runs |
|-------|-----------|
| `auto` | transformers' default: `grouped_mm`, or its per-expert loop where torch's grouped GEMM is missing |
| `grouped` | This library's forward: a stable sort of byte keys for the dispatch, torch's grouped GEMM, no masking passes; fp32 master weights are cast to the autocast dtype like `nn.Linear`'s (torch's grouped GEMM is not autocast-aware and otherwise computes in fp32) |
| `padded` | Every expert's tokens padded to the busiest expert's count, one batched GEMM per projection: no per-expert kernel launches, more memory when the load is unbalanced |
| `eager`, `grouped_mm`, `batched_mm`, ... | transformers' implementations by name |

torch's grouped GEMM has a fused kernel for bf16 on Hopper and newer GPUs; elsewhere (A100, CPU, fp32) it loops over
the experts inside the operator. Which value is fastest depends on the GPU, the token count and the balance, so
measure it:

```bash
python scripts/bench_moe_experts.py --experts 128 --top-k 8 --hidden 2048 --intermediate 768 --model
```

`grouped` and `padded` are tested equal to the eager experts (outputs and every gradient) for the gated, transposed
and biased layouts of the five families.

## Expert parallelism

Under FSDP every rank gathers each layer's full weights for its forward. In a MoE model that is mostly expert
weights, of which a token uses `top_k / E`. With `parallel.expert_parallel: true` every rank keeps `E / ranks` experts
of each layer in place and the tokens travel instead: each (token, expert) pair is sent to the rank that owns the
expert, computed there together with the other ranks' tokens, and returned (two all-to-all exchanges per layer, plus
one all-gather of the per-expert counts). The expert parameters are sharded on their expert dimension over the FSDP
mesh, so the optimizer, gradient clipping and checkpoints treat them like any other sharded parameter; everything
else stays under FSDP. An expert's weights receive the gradient of every token routed to it, from every rank.

Requirements and limits:

- `parallel.fsdp: true`, a number of ranks that divides the number of experts, no context parallelism, no SeCO, no
  CPU offload.
- `pgs train` only (not the RL and distillation trainers).
- One expert-parallel dimension over all ranks, each holding a contiguous block of experts. There is no placement
  by load and no replication of busy experts (DeepSeek's [EPLB](https://github.com/deepseek-ai/EPLB) does both, for
  inference): a rank's work follows the load of the experts it holds. `moe/rank_load_max` reports the imbalance
  between ranks, which is what a placement could remove; balanced routers keep it near 1.
- **Experimental.** Losses and gradients are tested equal to a single process on 2 and 4 CPU ranks, with the
  balancing losses, and a short training run reaches the same weights as FSDP alone. It has not been run on several
  GPUs, so its speed is unmeasured.

## Tiny models

`palingenesis.moe.tiny_moe(family)` builds a small randomly initialised model of a family with transformers' own
classes (4 layers, 8 experts, 2 per token): `qwen3_moe`, `qwen3_5_moe`, `mixtral`, `gpt_oss`, `deepseek_v3`. They are
what the tests run on, and are handy for ablations of the options above:

```python
from palingenesis.moe import RouterBalancer, find_moe_layers, tiny_moe

model = tiny_moe("deepseek_v3", num_hidden_layers=2)
balancer = RouterBalancer.attach(model)        # the defaults of this page
with balancer.batch(attention_mask):
    loss = model(input_ids=ids, labels=ids).loss
    balancer.sync()
    loss.backward()
balancer.step()                                 # after optimizer.step()
print(balancer.metrics())
```

## What is measured and what is not

Measured (CPU, tiny random models, in the test suite):

- The injected router gradients and the logged loss values equal the written-out losses, for the five families and
  the paths in the table above, including 2 and 3 ranks.
- `grouped` and `padded` expert forwards equal the eager experts; expert parallelism equals one process.
- Routing replay with the model's own routing changes nothing; with another model's routing every token runs through
  the recorded experts with gate values from the trained router.
- An exported model loads with transformers and has the layout of the checkpoint it started from.
- In a short run from a collapsed start, the balancing loss spreads the load again while the task loss alone
  does not.

Not measured:

- Any effect on the quality of a real fine-tune or RL run. The defaults follow the published coefficients; nothing
  here was ablated on a real MoE checkpoint.
- Expert parallelism on GPUs (throughput, multi-node), NVFP4 / MXFP8 expert kernels, models of real size.
- Routing replay against a real vLLM MoE deployment at scale. The numbers quoted for the mismatch are the cited
  papers', not ours.

## References

- Switch Transformer, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961): the load-balancing loss.
- GShard, [arXiv:2006.16668](https://arxiv.org/abs/2006.16668): the auxiliary loss it descends from.
- ST-MoE, [arXiv:2202.08906](https://arxiv.org/abs/2202.08906): router z-loss, fp32 routers, fine-tuning MoE models.
- DeepSeekMoE, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066), and DeepSeek-V3,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437): sigmoid routers, the selection bias and its rate.
- Auxiliary-loss-free load balancing, [arXiv:2408.15664](https://arxiv.org/abs/2408.15664): the bias update rule.
- Global-batch load balancing, [arXiv:2501.11873](https://arxiv.org/abs/2501.11873): the synchronised counts and the
  accumulation buffer.
- Rollout Routing Replay, [arXiv:2510.11370](https://arxiv.org/abs/2510.11370), and GSPO,
  [arXiv:2507.18071](https://arxiv.org/abs/2507.18071): routing consistency in RL.
- vLLM, [returning routed experts](https://docs.vllm.ai/en/latest/training/routed_experts/).
- torchtitan's MoE token dispatchers and expert parallelism, [github.com/pytorch/torchtitan](https://github.com/pytorch/torchtitan).
