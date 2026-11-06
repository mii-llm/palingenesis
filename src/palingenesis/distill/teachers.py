"""Online teachers: frozen causal LMs run on every micro-batch.

The teacher's forward runs without gradient and keeps only its final hidden states at the trained
positions (and the feature modules' outputs asked for); its logits are projected a slice of rows
at a time inside the loss (chunked.chunked_rows_loss), never as a whole.

Placement:
  replicate   a full copy on every rank's GPU (or `device`)
  shard       FSDP2-sharded over the data-parallel ranks like the student (a teacher larger than
              one GPU's share); its parameters are gathered layer by layer during its forward
  auto        shard when the student is sharded, replicate otherwise
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from palingenesis.distill.config import TeacherSpec
from palingenesis.logits import final_hidden_states, output_head, verify_output_head

logger = logging.getLogger(__name__)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class TeacherVocabError(ValueError):
    """The teacher's vocabulary cannot be compared with the student's token by token."""


def shared_vocabulary(student_tok, teacher_tok, spec: TeacherSpec):
    """The bridge between the student's and the teacher's vocabularies (opd.token_bridge): identical
    tokenizers, or a teacher vocabulary that is a prefix of the student's with an end-of-turn mapping.
    Raises TeacherVocabError for unrelated tokenizers: token-level KD needs the same ids for the same
    text (use on-policy distillation's xtok loss across tokenizers)."""
    from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError, check_compatible

    if teacher_tok is None or teacher_tok is student_tok:
        return TokenBridge(shared_vocab_size=len(student_tok))
    try:
        bridge = TokenBridge.from_tokenizers(student_tok, teacher_tok, spec.eos_map or None)
        check_compatible(student_tok, teacher_tok, bridge, spec.probe_texts)
    except TokenBridgeError as e:
        raise TeacherVocabError(
            f"distill.teachers.{spec.name}: the teacher's tokenizer does not share the student's vocabulary ({e}). "
            "Off-policy distillation compares distributions token by token, which needs the same token ids for "
            "the same text. Use a teacher of the same tokenizer family, or on-policy distillation (pgs distill) "
            "whose xtok loss works across tokenizers."
        ) from e
    return bridge


class OnlineTeacher:
    """A frozen teacher model, its output head and its vocabulary bridge."""

    def __init__(
        self,
        spec: TeacherSpec,
        student_tok,
        device: torch.device,
        attn_implementation: str,
        trust_remote_code: bool = True,
        mesh=None,
        parallel_config=None,
        bf16: bool = True,
        model: nn.Module | None = None,
        tokenizer=None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.name = spec.name
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer or spec.model, trust_remote_code=trust_remote_code)
        self.tokenizer = tokenizer
        self.bridge = shared_vocabulary(student_tok, tokenizer, spec)
        if model is None:
            logger.info("Loading teacher %s: %s (%s, frozen)", spec.name, spec.model, spec.dtype)
            model = AutoModelForCausalLM.from_pretrained(
                spec.model,
                dtype=DTYPES[spec.dtype],
                attn_implementation=spec.attn_implementation or attn_implementation,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        model.requires_grad_(False)
        model.eval()
        placement = spec.placement
        if placement == "auto":
            placement = "shard" if mesh is not None else "replicate"
        if placement == "shard" and mesh is None:
            logger.info("teacher %s: placement shard needs several ranks with FSDP; replicating", spec.name)
            placement = "replicate"
        self.placement = placement
        if placement == "shard":
            from palingenesis.config import ParallelConfig
            from palingenesis.distributed import apply_fsdp

            # the teacher is only read: parameters in its own dtype unless bf16 compute is asked
            model = apply_fsdp(model, mesh, parallel_config or ParallelConfig(), bf16=bf16)
            self.device = device
        else:
            self.device = torch.device(spec.device) if spec.device else device
            model = model.to(self.device)
        self.model = model
        self.head = output_head(model)
        if self.head is None:
            raise RuntimeError(f"teacher {spec.name}: cannot find its output head")
        verify_output_head(model, self.head, device=self.device)
        self.vocab_size = int(self.head.weight.shape[0]) if placement != "shard" else _full_rows(self.head)
        self.hidden_size = _hidden_size(model)

    @torch.no_grad()
    def hidden_states(self, input_ids: Tensor, forward_kwargs: dict, compute_dtype, autocast: bool) -> Tensor:
        """Final hidden states [B, T, H] on the student's device (no gradient)."""
        ids = input_ids.to(self.device)
        kwargs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in forward_kwargs.items() if k != "input_ids"
        }
        attention_mask = kwargs.pop("attention_mask", None)
        with torch.amp.autocast(ids.device.type, dtype=compute_dtype, enabled=autocast and ids.is_cuda):
            hidden = final_hidden_states(self.model, ids, attention_mask, **kwargs)
        return hidden

    def logits(self, hidden_rows: Tensor) -> Tensor:
        """Teacher logits (fp32) for hidden-state rows on its own device, returned where the rows were."""
        where = hidden_rows.device
        with torch.no_grad():
            out = self.head(hidden_rows.to(self.device)).float()
        return out.to(where)

    def release(self) -> None:
        """Reshard a sharded teacher's root parameters (kept gathered after its forward)."""
        if self.placement == "shard" and hasattr(self.model, "reshard"):
            self.model.reshard()


def _hidden_size(model: nn.Module) -> int:
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    return int(cfg.hidden_size)


def _full_rows(head: nn.Module) -> int:
    weight = head.weight
    return int(weight.full_tensor().shape[0]) if hasattr(weight, "full_tensor") else int(weight.shape[0])
