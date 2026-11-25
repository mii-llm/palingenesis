# Quantisation-Aware Training (FP4 / FP8)

*Train or distill a model so that it keeps its quality once its weights (and optionally activations) are stored in NVFP4, MXFP4 or FP8, then export a checkpoint vLLM serves.*

!!! tip "When to use this"
    - You will serve the model in NVFP4 / MXFP4 (Blackwell, or 4-bit weights on Ampere / Hopper) or FP8, and round-to-nearest quantisation (PTQ) loses too much.
    - The model is small (0.5B to a few B parameters): the fewer the parameters, the more a 4-bit grid costs.
    - You are distilling or fine-tuning anyway: QAT is the same run with fake quantisation in the forward.

    FP8 rarely needs QAT (its PTQ is close to lossless, see [Measured](#measured)). Measure first: `pgs quantize` writes the PTQ checkpoint in seconds.

## The two steps

```bash
# 1. PTQ baseline: how much does the format cost? (packed for vLLM + dequantized bf16 for any loader)
pgs quantize --model Qwen/Qwen3.5-0.8B --format nvfp4 --scales mse --out ptq-nvfp4 --dequantized ptq-nvfp4-bf16

# 2. If the gap is too large: QAT while distilling from the full-precision model
pgs train --config qat.yaml       # writes final/ (dequantized) and final-nvfp4/ (packed, vLLM)
vllm serve runs/qat/final-nvfp4
```

## Recipe: QAT with knowledge distillation

The recommended recipe is quantisation-aware distillation: the student is the model being quantised, the teacher is the same model at full precision, and the loss is the KL divergence between them (no labels needed). Matching the full-precision model's distribution recovers more than training the quantised model on the original labels, which also pulls it away from the behaviour it had (LLM-QAT, BitDistiller, NVIDIA ModelOpt's QAD; see [References](#references)).

```yaml
model:
  name_or_path: Qwen/Qwen3.5-0.8B
  torch_dtype: float32            # fp32 latent weights (bf16 compute with train.bf16): QAT's updates are small
data:
  dataset: data/self_generated.jsonl   # the model's own answers to varied prompts, or your SFT mix
  max_seq_length: 1024
train:
  output_dir: runs/qat
  max_steps: 1000
  per_device_batch_size: 8
  learning_rate: 1.0e-5           # about the SFT learning rate, or lower
  warmup_ratio: 0.05
  weight_decay: 0.0
distill:
  enabled: true
  temperature: 1.0
  ce_weight: 0.0                  # pure KL to the teacher
  teachers:
    bf16: {model: Qwen/Qwen3.5-0.8B, dtype: bfloat16}   # the full-precision model
plugins:
  custom:
    - class: palingenesis.qat:FloatQAT
      args:
        format: nvfp4             # nvfp4 | mxfp4 | fp8 | fp8_block
        activations: none         # weight-only; nvfp4 / mxfp4 / fp8: weights and activations
        scales: mse               # MSE-optimal clipping searched before the first step
        freeze_scales_at: 0.9     # the last 10% train on the exported grid
```

Notes on the recipe:

- **Data.** Self-generated text (the full-precision model's own sampled answers to diverse prompts) is a good default: it is in the teacher's distribution, needs no labels and covers what the model actually says (LLM-QAT). Mix in your SFT data if the model is also being fine-tuned.
- **Start from PTQ.** With `strength: 1` (the default) the first step is the PTQ model. `scales: mse` starts from the PTQ with the lower weight error; that is not always the better model (in the measurements below it lowers the weight error by about 1% and leaves the KL to the bf16 model unchanged for NVFP4, slightly worse for MXFP4), so compare both with `pgs quantize` if the starting point matters. A ramp (`strength: {start: 0.0, end: 0.1}`) eases the switch for a model that is also learning a new task.
- **Learning rate.** QAT moves the latent weights by a few quantisation steps at most; 1e-5 for full weights of a 0.5–2B model is a reasonable start. Keep fp32 latent weights (`model.torch_dtype: float32`): bf16 weights round most such updates away.
- **Freeze.** `freeze_scales_at: 0.9` fixes the scales (no moving average: computed once from the weights at that step), so the last steps settle the weights on the grid that is exported.
- **On-policy distillation and RL.** The same plugin runs in `pgs distill` and `pgs rl` (`plugins: {custom: [...]}` in their configs): the rollout engine receives the quantised weights at every push, so the student samples from the model being trained. Start with off-policy KD in `pgs train`; OPD after it fixes what the quantised model does on its own samples.

What to watch (`plugins/float_qat/*`, every `metrics_every` steps): `qerr_mean` / `qerr_max` (relative squared weight error per layer), `saturated_blocks` (share of blocks with a clipped value: high after a freeze means the weights moved past their frozen scales), `clip_mean` (`scales: mse | learned`), `act_amax_max` (NVFP4 activations). The distillation loss (`train/kd_loss`) falling is the recovery itself.

### Measured

PTQ of Qwen3.5-0.8B (weight-only, `pgs quantize`), one A100 80GB, `scripts/gpu_checks/fp4_qat.sh`: KL(bf16 ‖ quantised) per token and top-1 agreement over 42k tokens of held-out self-generated text, accuracy on 200 GSM8K test items (greedy, non-thinking; bf16: 0.51; one standard error ≈ 0.035), relative squared weight error averaged over the 150 quantised layers.

| Format, scales | KL (nats) | Top-1 agreement | Accuracy | Weight error |
|---|---|---|---|---|
| `nvfp4`, `max` | 0.142 | 0.812 | 0.455 | 0.0090 |
| `nvfp4`, `mse` | 0.141 | 0.814 | 0.410 | 0.0089 |
| `mxfp4`, `max` | 0.221 | 0.769 | 0.350 | 0.0130 |
| `mxfp4`, `mse` | 0.232 | 0.771 | 0.355 | 0.0128 |
| `fp8`, `max` | 0.0092 | 0.946 | 0.530 | 0.00070 |
| `fp8_block`, `max` | 0.0091 | 0.948 | 0.535 | 0.00070 |

So on a model this small, FP4 PTQ is far from free (NVFP4 loses about 5 points of the proxy, MXFP4 about 16, and a fifth of the next-token choices change), FP8 PTQ is within noise, and the dequantized export computes exactly what the fake-quantised forward computes (maximum logit difference 0). The packed NVFP4 and MXFP4 checkpoints load in vLLM 0.26 on the A100 (accuracy 0.415 and 0.365, within noise of their dequantized twins' 0.455 and 0.350).

**QAT recovery is not measured yet** (an estimate from the literature: a few hundred to a few thousand steps of QAT with KD close half or more of the KL gap). The same script trains NVFP4 and MXFP4 with the recipe above and reports the KL and accuracy next to PTQ.

## Formats

| `format` | Elements | Scales | Block | Native hardware | vLLM on A100 / H100 | Checkpoint |
|---|---|---|---|---|---|---|
| `nvfp4` | E2M1 (FP4) | E4M3 per block + fp32 per tensor | 16 | Blackwell (W4A4) | weight-only (W4A16, Marlin) | compressed-tensors `nvfp4-pack-quantized` |
| `mxfp4` | E2M1 (FP4) | E8M0 (power of two) | 32 | Blackwell (W4A4) | weight-only (W4A16, Marlin) | compressed-tensors `mxfp4-pack-quantized` |
| `fp8` | E4M3 | fp32 per output row | row | Hopper, Ada, Blackwell | weight-only (W8A16: Humming, else Marlin) below CC 8.9; W8A8 from CC 8.9 when trained with `activations: fp8` | compressed-tensors `float-quantized` (channel) |
| `fp8_block` | E4M3 | fp32 per 128 × 128 tile | 128 × 128 | Hopper, Blackwell | weight-only below CC 8.9; W8A8 from CC 8.9 (always: see below) | compressed-tensors `float-quantized` (block + dynamic inputs) |

Definitions (palingenesis.qat.formats; the conventions of compressed-tensors, vLLM and torchao):

- **E2M1** values ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, round to nearest even, saturating at 6 (OCP MX v1.0).
- **NVFP4**: `global_scale = 448·6 / amax(tensor)` (stored as a divisor), `block_scale = e4m3(clamp(amax(block)/6 · global_scale, 2⁻⁶, 448))`, `W ≈ q · block_scale / global_scale`. The lower clamp keeps vLLM's Marlin kernel (which zeroes smaller block scales) and the training forward identical. Layers an engine fuses into one matmul (q/k/v, gate/up, the Gated DeltaNet input projections) share the per-tensor scale, as vLLM keeps one per fused layer (`fused_groups`).
- **MXFP4**: `scale = 2^e` with `e` from the block's amax; `mx_rounding`: `floor` (the OCP reference, may clip the block max), `even` (default: compressed-tensors' and torchao's rule, rounds up from a significand of 1.75), `ceil` (never clips).
- **FP8**: `scale = amax / 448` per row or per 128 × 128 tile, E4M3 with round to nearest even.
- Non-finite weights give NaN outputs (they are never quantised away); an all-zero block quantises to zeros.

### Weight-only and weights + activations

`activations: none` (default) quantises the weights only: what every GPU can serve. `activations: fp8 | nvfp4 | mxfp4` also quantises each linear layer's input, per token, with dynamic scales (`fp8` with `fp8_block` weights: per token and 128 inputs, as vLLM's FP8 block scheme). NVFP4 activations have a static per-tensor scale at inference (`input_global_scale`): the plugin records the largest input seen (all ranks), uses it from the freeze on, and exports it. Activation formats run as trained only on hardware that has them (FP8: Hopper / Ada; FP4: Blackwell); on an A100, vLLM runs such a checkpoint with bf16 activations.

`fp8_block` is the exception to "weight-only is what every GPU serves as trained": vLLM (0.26) loads a block-scaled FP8 checkpoint only with its dynamic FP8 input scheme (compressed-tensors' `FP8_BLOCK` preset), so the packed checkpoint always carries it. A GPU without FP8 matmuls (compute capability below 8.9) ignores it and runs weight-only; Ada, Hopper and Blackwell also quantise each layer's input. To serve there exactly as trained, train `fp8_block` with `activations: fp8`, or use `format: fp8`, whose weight-only checkpoint stays weight-only everywhere. The plugin logs this at startup.

### Scales

| `scales` | What it does |
|---|---|
| `max` | each block's scale from its largest value (round-to-nearest PTQ) |
| `mse` | a clipping factor per output row (per 128 rows for `fp8_block`), searched over `clip_steps` values from 1 to `clip_min` to minimise the row's squared error; fixed afterwards |
| `learned` | the same factors, initialised by the search, trained with LSQ gradients in their own optimizer group (`clip_lr`) and kept in [`clip_min`, 1] by projection; `pgs train` and `pgs distill` only |

## Export

Every export (`final/`, `best/model/`) of a run with the plugin gives two models:

- **`final/`**: the dequantized model (`export: dequantized`, default), in the original architecture and dtype. Its weights are exactly what the training forward computed with (the exported tensors equal the fake-quantised ones bit for bit; an fp32 run's export, loaded in bf16, gives the bf16 weights its autocast forward used), so it evaluates the quantised model on any hardware and in any loader. `export: latent` writes the latent weights instead (to continue training elsewhere; learned clipping factors are not kept).
- **`final-<format>/`**: the packed checkpoint in the compressed-tensors format (`quant_method: compressed-tensors`), with the tokenizer and config of `final/`. `vllm serve final-nvfp4` loads it. Layers left in full precision (the head, the embedding, excluded layers, shapes the block does not divide, a vision tower) are listed in `ignore`; a `quantization_report.json` holds each layer's error. Written for the format pairs vLLM has a scheme for: `nvfp4` / `mxfp4` with `none` or their own activations, `fp8` / `fp8_block` with `none` or `fp8` (`quantized_export: auto`).

compressed-tensors is chosen over ModelOpt's format because it is the one vLLM loads for all four formats here (ModelOpt has NVFP4 and FP8 but no dense MXFP4); both run NVFP4 through the same kernels.

`pgs quantize` is the PTQ path with the same code: `--format`, `--activations`, `--scales max|mse`, `--mx_rounding`, `--exclude`, `--calibration FILE` (lines of text; needed for NVFP4 activations), `--out` (packed), `--dequantized` (bf16).

## Plugin arguments

| Argument | Default | Meaning |
|---|---|---|
| `format` | `nvfp4` | `nvfp4`, `mxfp4`, `fp8`, `fp8_block` |
| `activations` | `none` | `none`, `fp8`, `nvfp4`, `mxfp4` (dynamic, per token) |
| `include` | `["(^\|\.)layers\.\d+\."]` | regular expressions of the `nn.Linear` names to quantise (default: the decoder layers: attention, MLP, Gated DeltaNet projections; not a vision tower) |
| `exclude` | `["\.in_proj_[ab]$", "\.mlp\.gate$", "\.shared_expert_gate$", "(^\|\.)(visual\|vision_tower\|vision_model\|audio_tower\|audio_model\|multi_modal_projector)\."]` | regular expressions of names kept in full precision (default: Gated DeltaNet's per-head decay / gate projections, too narrow for the 4-bit kernels and numerically sensitive, MoE routers, and vision / audio encoders, which an engine cannot load packed); a list replaces the default |
| `quantize_embedding` / `quantize_head` | `false` | also fake-quantise the embedding / output head (exported dequantized in both checkpoints: vLLM keeps them in bf16) |
| `scales` | `max` | `max`, `mse`, `learned` |
| `clip_min`, `clip_steps` | `0.5`, `16` | the clipping search range and resolution; the learned factors stay in [`clip_min`, 1] |
| `clip_lr` | `1e-3` | learning rate of the learned factors (0: `train.learning_rate`) |
| `mx_rounding` | `even` | MXFP4 exponent rounding: `floor`, `even`, `ceil` |
| `strength` | `1.0` | quantisation strength, a number or a ramp `{start, end, start_value, end_value, shape}` over training progress: `W + strength · (W_q − W)` |
| `freeze_scales_at` | `null` | fraction of training at which the scales are fixed |
| `fused_groups` | q/k/v, gate/up, `in_proj_qkv`/`in_proj_z`, `in_proj_b`/`in_proj_a` | sibling layers that share NVFP4's per-tensor scale and are quantised all or none |
| `export` | `dequantized` | `dequantized` or `latent` (the model in `final/`) |
| `quantized_export` | `auto` | write `final-<format>/` (`auto`: when vLLM has a scheme for the pair; `true`: error otherwise) |
| `layer_metrics` | `false` | also log `qerr/<layer>` and `saturated/<layer>` |
| `metrics_every` | `50` | steps between the (gathering) error metrics |

Layers whose shape the block does not divide stay in full precision (logged at startup, listed in `ignore`). With FSDP the forward sees whole weights, so per-tensor and fused scales are exact; the metrics, the clipping search and the rollout push gather each weight once. The fake quantisation computes in float32 next to each matmul: a step-time overhead (forward + backward, Qwen3.5-0.8B, bf16, 4 × 1024 tokens on an A100, against 0.204 s plain: `fp8` / `fp8_block` 1.23×, `mxfp4` 1.40×, `nvfp4` 1.49×, `fp8` with FP8 activations 1.44×, `nvfp4` with NVFP4 activations 1.98×), and slow HF-backend generation (vLLM rollouts load the dequantized weights instead).

## Hardware notes

- **Training** needs no low-precision hardware: the forward fake-quantises in float32 and computes in bf16.
- **A100 / Ampere** (compute capability 7.5 to 8.8): vLLM serves NVFP4 and MXFP4 weights (Marlin, 4-bit weights and bf16 activations: memory and bandwidth savings; measured loading on an A100 with vLLM 0.26) and FP8 weights (W8A16). No FP4 / FP8 matmuls. For FP8 weights vLLM 0.26 picks its Humming kernel first, which it compiles at the first forward with NVRTC, then Marlin. If the engine dies there (`humming.jit.compiler.NVRTCCompiler ... run failed`, seen on an A100 box), serve through Marlin: `VLLM_DISABLED_KERNELS=HummingFP8ScaledMMLinearKernel vllm serve final-fp8` (or `--linear-backend marlin`). The GPU check retries that way and reports which kernels served.
- **H100 / Ada**: FP8 W8A8 natively; FP4 weights through Marlin.
- **Blackwell (B200, RTX 50)**: NVFP4 and MXFP4 W4A4 natively (CUTLASS).

Rotations before quantisation (Hadamard transforms as in HALO, QuaRot or Quartet) spread outliers and help FP4 further, but the inference engine must apply the same rotation online; the checkpoints written here do not carry one, so the plugin does not rotate. Block scales of 16 / 32 values already isolate most outliers.

## References

- Bengio, Léonard, Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv:1308.3432 (straight-through estimator).
- Esser et al. *Learned Step Size Quantization.* ICLR 2020, arXiv:1902.08153 (LSQ).
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
- Liu et al. *LLM-QAT: Data-Free Quantization Aware Training for Large Language Models.* arXiv:2305.17888 (KD from the full-precision model on self-generated data).
- Du et al. *BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation.* arXiv:2402.10631.
- Chen et al. *EfficientQAT: Efficient Quantization-Aware Training for Large Language Models.* arXiv:2407.11062.
- NVIDIA TensorRT Model Optimizer (Apache-2.0): QAT and quantisation-aware distillation (QAD) guides and the NVFP4 checkpoint format; NVIDIA, *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference* (2025); NVIDIA, *Pretraining Large Language Models with NVFP4*, arXiv:2509.25149.
- Open Compute Project. *OCP Microscaling Formats (MX) Specification v1.0* (2023); Rouhani et al. *Microscaling Data Formats for Deep Learning*, arXiv:2310.10537.
- Micikevicius et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433; DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437 (128 × 128 block-scaled FP8).
- PyTorch torchao (BSD-3-Clause): `torchao.quantization.qat` and `torchao.prototype.mx_formats` (MX scale rounding modes, NVFP4 scale clamping).
- vLLM project, compressed-tensors and llm-compressor (Apache-2.0): the checkpoint format and its loaders.
- Ashkboos et al. *HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs.* arXiv:2501.02625.

The implementation follows these specifications and papers; no code was copied from the projects above.
