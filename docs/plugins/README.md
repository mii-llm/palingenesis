# Plugins Reference

Research papers backing each plugin implementation. Each plugin has been
independently validated and composes with the full optimization stack.

## SymNoise (Symmetric Noisy Embeddings)

**Papers:**
- "Advancing Language Model Fine-tuning with Symmetric Noise" (arxiv:2312.01523, Dec 2023)
- "Understanding and Improving Noisy Embedding Techniques" (arxiv:2605.23171, May 2025)

**Key result:** +6.7% over NEFTune on AlpacaEval (69.04% vs 64.69% vs 29.79% baseline)

**Implementation notes:**
- The full SymNoise algorithm (Algorithm 2) concatenates BOTH x+noise and x-noise into one batch
- This doubles effective batch size and requires duplicating labels
- Our default uses the simplified Bernoulli variant which is compatible with FSDP/CP/compile
- The key difference from NEFTune: Bernoulli {-1,+1} noise instead of Uniform(-1,1)
- Noise scaling: alpha / sqrt(seq_len * embed_dim), default alpha=5

**Config:** `plugins.sym_noise: true`, `plugins.sym_noise_alpha: 5.0`

## InfoSFT (Information-Aware Token Weighting)

**Paper:** "Learn More and Forget Less with Information-Aware Token Weighting"
(arxiv:2605.14967, Sabbaghi et al., UPenn/USC, May 2025)

**Key result:** Consistent gains over SFT and DFT across math, code, CoT.
Better learning-forgetting tradeoff (less catastrophic forgetting).

**Implementation (exact formula from Eq. InfoSFT_grad):**
```
w(q) = q * [logit(p_bar) - logit(q)]_+
```
where:
- `q = pi_theta(y_t | x, y_<t)` -- model's probability of the correct token
- `p_bar = 0.93` -- calibration constant (empirically stable across models)
- `logit(x) = log(x/(1-x))`
- `[x]_+ = max(x, 0)`

**Weighting profile:**
- q -> 1 (high confidence): logit(q) > logit(p_bar), so correction is 0
- q -> 0 (too surprising): weight -> 0 (the q factor)
- q ~ 0.3-0.8 (medium): highest weight (most informative)

**Practical notes:**
- Default p_bar=0.93 works across all models tested (range [0.9, 0.95])
- One extra softmax+gather per forward (negligible cost)
- For CoT/reasoning data where format is very novel, use SFT for 1 epoch first
  then switch to InfoSFT (paper shows this is complementary)

**Config:** `plugins.info_sft: true`, `plugins.info_sft_temperature: 2.0`

## Schedule-Free AdamW

**Papers:**
- "The Road Less Scheduled" (Defazio et al., ICLR 2024)
- "Through the River: Understanding the Benefit of Schedule-Free Methods" (NeurIPS 2025)

**Key result:** Matches cosine schedule without knowing total steps. Navigates
loss landscape "river structure" via iterate averaging.

**Implementation notes:**
- Uses `schedulefree` library from Facebook Research
- Has special train/eval mode: call optimizer.train() / optimizer.eval()
- No LR scheduler needed (set scheduler=None in training loop)
- Same memory footprint as standard AdamW

**When to use:**
- Streaming datasets where total size is unknown
- Exploratory training where you want to stop early
- Avoids the "restart with different max_steps" problem

**Config:** `plugins.schedule_free: true`


## Pre-RL Mode (SFT for RL Warm-Start)

**Paper**: "EKSFT" (arxiv:2605.29303, May 2026)

**Key result**: SFT that preserves output diversity for subsequent GRPO/DPO/PPO.
Standard SFT sharpens the policy distribution, making RL exploration impossible.
Pre-RL mode keeps the distribution broad.

**Mechanism**:
1. Compute per-token entropy and KL from the model's own pre-update distribution
2. MASK tokens with high entropy OR high KL (unsafe to force-imitate)
3. CE loss only on UNMASKED tokens (safe: medium confidence, low drift)
4. Entropy BONUS on masked tokens (keep them diverse for RL sampling)
5. KL PENALTY on masked tokens (don't drift from base on uncertain positions)

**When to use**: Your pipeline is `SFT -> GRPO/DPO` with a verifier/reward model.
The model learns the task format (tool calls, etc.) while retaining the sampling
diversity that RL needs to explore different solutions.

**Config:**
```yaml
plugins:
  pre_rl: true
  pre_rl_entropy_coeff: 0.1    # Higher = more diverse outputs
  pre_rl_kl_coeff: 0.5         # Higher = less drift from base
```

**Mutually exclusive with**: `info_sft` (which optimizes for final SFT quality, not RL readiness)

**Overhead**: ~15% (extra no_grad forward for reference distribution). Compiled via torch.compile.

**Post-SFT workflow**:
```bash
# Step 1: Pre-RL SFT (learn format, preserve diversity)
pgs train --config sft_config.yaml --plugins.pre_rl true

# Step 2: GRPO/DPO (use TRL or your RL framework)
# The model from step 1 is an excellent RL starting point
```
