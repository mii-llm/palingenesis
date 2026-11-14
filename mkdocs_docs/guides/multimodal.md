# Multimodal Training (Images)

*Images in SFT, DPO, off-policy distillation and multi-turn RL, for the Qwen-VL family (`palingenesis.multimodal`).*

A vision-language model reads an image as a run of placeholder tokens whose embeddings are replaced by the vision tower's output. Training on images is exact only when three things agree with inference:

- the **number of placeholder tokens** of each image, which the image processor's resize decides;
- the **pixels**, handed to the vision tower in the order the placeholders appear;
- the **position** of every token. Qwen-VL models use multimodal RoPE (M-RoPE): a text token advances the temporal, height and width axes together; an image of $h \times w$ tokens keeps one temporal index, spreads over height and width, and the text after it resumes at $\text{start} + \max(h, w)$.

With `multimodal.enabled: true` the trainer loads the checkpoint in its multimodal architecture, runs the checkpoint's own image processor in the data pipeline, and computes the positions per document, so packed rows carry exactly the positions each document has on its own.

!!! abstract "Use it when"
    Your data has images (screenshots, documents, photos) and the model is a Qwen2-VL, Qwen2.5-VL, Qwen3-VL or Qwen3.5 checkpoint. Without it, such checkpoints train as text models: image parts in the data are ignored (a warning says so) and exports copy the vision tower unchanged.

## Minimal setup

```yaml
model:
  name_or_path: Qwen/Qwen3.5-2B
multimodal:
  enabled: true
  train_vision_tower: false     # the vision encoder
  train_projector: false        # the patch merger that maps vision features to the language model
  max_pixels: 1048576           # at most 1024 tokens per image with 16 px patches and a 2 x 2 merge
data:
  dataset: your-org/screens
  max_seq_length: 8192          # image tokens count against it
```

It needs `torchvision` and `pillow` (the image processors of transformers run on torchvision; both come with the `vllm` extra). Without torchvision the Qwen processor's pillow backend is used, with a warning: the token counts are the same, the pixels differ in the last bits of the interpolation.

## Data format

Images are content parts of a message, in any role the chat template renders them in:

```json
{"messages": [
  {"role": "user", "content": [
    {"type": "text", "text": "What does the dialog say?"},
    {"type": "image", "image": "screens/0001.png"}]},
  {"role": "assistant", "content": [{"type": "text", "text": "It asks to save the file."}]}
]}
```

- The image is a local path, a `datasets` Image (`{"bytes": ..., "path": ...}`), raw bytes, a PIL image or a base64 `data:` URL. `image_url` (OpenAI) and `input_image` (Responses) parts are read the same way. Remote URLs are refused: a data worker does not download.
- In JSONL and Parquet every message of a column must have the same shape, so write text as parts too (`[{"type": "text", "text": ...}]`). A message with only text parts is read as its text. Arrow fills absent keys with nulls (`"image": null` on a text part); those are dropped before the template sees them.
- Video parts are refused (not implemented).

## What the pipeline does

1. **Template.** The image parts are replaced by bare `{"type": "image"}` parts and the chat template renders one placeholder per image (`<|vision_start|><|image_pad|><|vision_end|>` for Qwen). The count is checked: a template that does not render an image in some role raises an error instead of training on a conversation with a missing picture.
2. **Expansion.** Each placeholder is repeated to its image's token count, `(pixels after resize / patch²) / merge²`, before tokenizing. Truncation, the masker and length-grouped batching therefore see the real sequence.
3. **Masking.** Image tokens are never targets, in every masking mode (assistant-only, `include_observations`, `last_turn_only`).
4. **Truncation.** Conversations are cut at turn boundaries as usual, so an image is kept whole or not at all. A row that would be cut inside an image is dropped (`dropped_image_cut` in the data statistics).
5. **Positions.** The M-RoPE positions are computed per document and travel with the batch (`mm_position_ids`), instead of being derived by the model from the padded batch.
6. **Forward.** `pixel_values` and `image_grid_thw` are concatenated in row order, which is the order the model fills the placeholders in.

### Packing

`data.packing: true` is supported. A packed row holds several documents and their images. The forward receives positions of shape `[4, B, S]`: the first row restarts at 0 at each document and is what transformers builds the block-diagonal attention mask from (and what resets linear-attention state); the other three are each document's own M-RoPE positions. The test suite checks that every document of a packed row gets the logits it has alone. A document with images that is longer than the packed block is an error (it cannot be cut).

### Vision tower and projector

| Setting | Vision encoder | Projector | Typical use |
|---|---|---|---|
| both `false` (default) | frozen | frozen | Language-side adaptation: agents, instruction following on screenshots |
| `train_projector: true` | frozen | trained | A new visual domain with few images |
| both `true` | trained | trained | Large in-domain image data |

Frozen modules still run on every image, take no optimizer state and keep no activations for the backward. Under FSDP each vision block is its own shard, like a language-model layer.

### Memory

Image tokens are ordinary tokens for the language model, so `data.max_seq_length` bounds them. `multimodal.max_pixels` bounds the tokens of one image. `pgs profile` adds the vision tower: its parameters (in the weights; in the optimizer and gradients only when trained) and a worst-case line for its pixels and activations, as if every token were an image token.

## What is supported

| Feature | With images |
|---|---|
| SFT, multi-source mixes, evaluation, per-source evaluation | Supported |
| Packing (sdpa, flash, linear-attention kernels) | Supported |
| Chunked loss, Cut Cross-Entropy, gated objectives | Supported (they act on hidden states) |
| FSDP | Supported (vision blocks sharded); CPU tests cover layer discovery only |
| DPO | Supported: both sides of a pair and the frozen reference read the pixels |
| Off-policy distillation, online teacher | Supported: the teacher is loaded in its multimodal architecture and must read images as the student's processor writes them (same image token, patch and merge sizes) |
| Off-policy distillation, cached teacher | Refused: the cache is keyed by tokens, and the same tokens with other pixels have other targets |
| Teacher views (context distillation) | Refused |
| SeCO (`memory.seco`), and with it loop recovery and the unlikelihood terms | Refused: the chunked forward runs the language model on token ids |
| Context parallel | Refused: an image cannot be split across the ranks of a sequence |
| Pretokenized cache (`data.pretokenize`) | Refused: it stores tokens, not pixels |
| RL (`pgs rl`) | Supported with the `hf` and in-process `vllm` engines; see below |
| On-policy distillation (`pgs distill`) | Refused |
| Video | Refused |
| Other families (Gemma 3, Llama 4, GLM-4.xV, Kimi-VL, ...) | Refused: their image layouts are not implemented |

## Exports

The model is saved in the checkpoint's own layout (language model, vision tower, projector) with the tokenizer and the image processor files, including the pixel bounds used in training. The directory loads with `AutoModelForImageTextToText` and is what vLLM serves.

## Multi-turn RL with images

```yaml
model:
  policy: Qwen/Qwen3.5-2B
  multimodal: true
  max_pixels: 1048576
env:
  type: my_envs.py:ComputerEnv
  image_results: tool          # or: user
  max_images: 3                # keep the last 3 screenshots in the context
```

A tool returns images next to its text:

```python
from palingenesis.rl.env import ToolOutput

class ComputerEnv:
    def click(self, x: int, y: int) -> ToolOutput:
        """Click at a screen position.

        Args:
            x: Horizontal pixel.
            y: Vertical pixel.
        """
        self.screen.click(x, y)
        return ToolOutput("clicked", images=[self.screen.screenshot()])
```

Prompts may carry image parts as in SFT.

### Where a tool's image goes

OpenAI-style chat completions take images only in user messages: the tool message is text. Anthropic's `tool_result` and the Responses API's `function_call_output` carry images inside the tool result. Agent harnesses follow the API they speak, so both layouts exist in deployment, and a model should be trained in the one it will be served in.

| `env.image_results` | Layout | Who uses it |
|---|---|---|
| `tool` (default) | The tool message holds its text, then the image | Anthropic `tool_result` (the computer-use reference loop), Responses `function_call_output` (Codex CLI's `view_image`, the OpenAI CUA sample), opencode on those protocols, OpenHands, SWE-agent |
| `user` | The tool message keeps its text; one user message after the turn's tool results holds the images | opencode and qwen-code on OpenAI-style chat completions, Qwen-Agent |

`env.image_tool_note` is the tool message's text when the tool returned only images, and `env.image_user_note` opens the user message (qwen-code: `[media attached in following user message]` and `(attached media from previous tool call)`; opencode sends neither). A harness profile sets its own layout with `HarnessProfile.image_results`; text-protocol environments answer in a user message, and the images follow its text.

Whether `tool` works depends on the chat template:

| Template | Image inside a tool message |
|---|---|
| Qwen3.5, Qwen3-VL | Rendered, inside `<tool_response>` in the user turn the template wraps tool results in |
| Qwen2.5-VL | No tool branch: the message renders as a `tool` turn the model was not trained on |
| Gemma 3, Pixtral | Tool role refused by the template |
| Llama 4, GLM-4.5V, Mistral Small 3.2 | Text only: the image is dropped or stringified |
| GLM-4.1V | Tool messages dropped |

The rollout checks the placeholder count after every tool result. A template that drops the image ends the trajectory with an error that names `env.image_results: user`.

### Rollout and training on the same tokens

Trajectories are token-native: tokens are appended as sampled and never re-tokenized. With images:

- The trajectory stores **one placeholder token per image** and the raw images, in order. This is what vLLM takes (`prompt_token_ids` with `multi_modal_data`; it expands the placeholders with the model's processor) and what the HF engine expands itself.
- The train step expands the same tokens with the same processor and bounds, computes the positions, and processes the pixels again. An image shared by a group's rollouts is processed once per micro-batch.
- Context limits (`rollout.max_model_len`) and micro-batch sizes count expanded tokens.
- A sampled image placeholder is an error: the trajectory is left out of training.

With the HF engine the test suite checks the result end to end: on fresh rollouts over several turns with new screenshots, the trainer's log-probabilities of the sampled tokens equal the sampler's (importance ratios of 1), and they stop being equal when the pixels are swapped.

### Dropping old screenshots

`env.max_images: N` keeps the last N images in the context. Older ones are replaced by `env.image_placeholder` (the placeholder token and the template's wrapper tokens go, the text takes their place; the tool's own text stays). Nothing else is re-rendered.

Dropping an image changes tokens the policy has already read, so the sequence is no longer append-only. Each rewrite closes a **segment**: the context as it was and the turns sampled on it. Every sampled token belongs to one segment and is trained on the context it was sampled from, with the trajectory's reward and advantage. A trajectory with $k$ rewrites trains as $k + 1$ sequences, so the train-step cost grows with the number of rewrites.

`env.image_prune_slack: S` lets S extra images accumulate before dropping back to N: one rewrite every $S + 1$ images instead of one per image. Fewer rewrites mean fewer training sequences and a longer-lived prefix cache in the engine. This is what harnesses do for their prompt cache (browser-use drops from 20 screenshots to 2; Anthropic's reference loop removes images in chunks).

| Harness | Screenshots kept | Placeholder |
|---|---|---|
| UI-TARS | last 5 | image turns removed |
| Anthropic computer-use reference loop | last 3 in its UI, removed in chunks | none |
| OSWorld baseline agent | last 3 turns | turns removed |
| browser-use | newest 2 once 20 have accumulated | `[older screenshot removed]` |
| Agent S3 | latest only, text kept | none |
| SWE-agent | last n observations | `(K images omitted)` |

Sequence-level objectives (`gspo`, the sequence mask) treat each segment as a sequence.

### Engines

| `rollout.backend` | Images |
|---|---|
| `hf` | Supported and tested on CPU |
| `vllm` (in-process) | Wired: images go as `multi_modal_data` next to un-expanded tokens and the processor bounds are passed as `mm_processor_kwargs`. Not yet run against vLLM |
| `vllm_server` | Refused: the server is driven through `/v1/completions`, which takes no images |

`loss.kl_ref` and `train.prompt_sharing: on` are refused with a multimodal policy.

## What is verified

Everything below runs on tiny randomly initialised Qwen3.5 models on the CPU (`tests/test_multimodal*.py`).

- M-RoPE positions equal those of transformers' `get_rope_index` for Qwen2-VL, Qwen2.5-VL, Qwen3-VL and Qwen3.5.
- A batch forward with our positions and pixel order equals the model's own multimodal forward; every document of a packed row gets the logits it has alone.
- Image tokens are never trained; truncation keeps whole images.
- The freeze options change exactly the weights they name; exports reload as the multimodal architecture with their processor.
- DPO and online-teacher distillation read the pixels on every model involved.
- RL: tool images in both layouts, token-identical rollout and training with the HF engine, context rewrites.

With the real Qwen3.5 tokenizer, chat template and image processor (no weights), the pipeline's token ids, image grids and pixel values equal `AutoProcessor`'s for a conversation with an image in the user turn and one in a tool result.

Not verified yet: training on a GPU, real checkpoints, FSDP runs (beyond layer discovery), the flash and linear-attention packed kernels with images, the in-process vLLM engine with images, and serving an export with vLLM.

## Sources

- Positions: transformers `Qwen2VLModel.get_rope_index` and its Qwen2.5-VL, Qwen3-VL and Qwen3.5 counterparts; Qwen2-VL ([arXiv:2409.12191](https://arxiv.org/abs/2409.12191)), Qwen2.5-VL ([arXiv:2502.13923](https://arxiv.org/abs/2502.13923)), Qwen3-VL ([arXiv:2511.21631](https://arxiv.org/abs/2511.21631)).
- Chat templates: the `chat_template` files of the checkpoints named in the table, read from the Hugging Face Hub.
- vLLM: `vllm/inputs` (tokens prompts with `multi_modal_data`), `vllm/multimodal/processing` (placeholder expansion on token ids), `vllm/entrypoints/openai/completion` (no images).
- APIs: the OpenAI Python SDK's `ChatCompletionToolMessageParam` (text only) and `ResponseFunctionCallOutputItemListParam` (text, image, file); Anthropic's tool-use documentation (image blocks in `tool_result`).
- Harnesses, read in their source: opencode (`packages/ai/src/protocols/openai-chat.ts`), qwen-code (`openaiContentGenerator/converter.ts`), Codex CLI (`core/src/tools/handlers/view_image.rs`), Anthropic's computer-use demo (`computer_use_demo/loop.py`), OpenHands (`openhands/sdk/llm/message.py`), SWE-agent (`sweagent/agent/history_processors.py`), browser-use (`browser_use/agent/loop.py`), UI-TARS-desktop (`packages/ui-tars/sdk`), OSWorld (`mm_agents/agent.py`), Agent S3 (`gui_agents/s3/agents/worker.py`), Qwen-Agent (`qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py`).
- Training frameworks with the same token-in, token-out design (stored rollout ids, raw images next to them, pixels processed again at training time): TRL's `GRPOTrainer`, verl's tool agent loop, OpenRLHF, SkyRL, slime, AReaL. None of them drops images inside a trajectory; per-step training on the context each step saw is the design of the GUI-agent RL papers: UI-TARS ([arXiv:2501.12326](https://arxiv.org/abs/2501.12326)), ComputerRL ([arXiv:2508.14040](https://arxiv.org/abs/2508.14040)), DART-GUI ([arXiv:2509.23866](https://arxiv.org/abs/2509.23866)).
