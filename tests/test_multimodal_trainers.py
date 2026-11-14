"""Images in DPO and in off-policy distillation with an online teacher: both policies / models read
the same pixels. Tiny randomly initialised Qwen3.5 models on the CPU."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from distill_helpers import cpu_config, run_cpu, save_model
from multimodal_helpers import save_tiny, tiny_image, tiny_tokenizer
from test_multimodal import _write_image_rows

from palingenesis.config import ConfigError
from palingenesis.multimodal import POSITION_FIELD, MultimodalError


def _write_pairs(tmp_path, n: int = 12) -> str:
    path = tmp_path / "pairs.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            file = tmp_path / f"pair_{i}.png"
            tiny_image(16 + 8 * (i % 3), 24, i).save(file)
            row = {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Describe picture {i}."},
                            {"type": "image", "image": str(file)},
                        ],
                    }
                ],
                "chosen": f"A small picture, number {i}.",
                "rejected": f"I cannot see pictures. I cannot see pictures. Number {i + 1}?",
            }
            f.write(json.dumps(row) + "\n")
    return str(path)


def test_dpo_with_images(tmp_path, monkeypatch):
    import palingenesis.dpo as dpo

    model_dir = save_tiny(tmp_path / "model")
    config = cpu_config(tmp_path, model_dir, _write_pairs(tmp_path), steps=3)
    config.multimodal.enabled = True
    config.dpo.enabled = True
    config.dpo.sft_weight = 0.1
    config.memory.chunked_loss = True
    config.data.eval_dataset = config.data.dataset
    config.data.eval_split = "train"
    config.data.eval_samples = 4
    config.data.eval_every = 2

    seen = []
    reference_logps = dpo.reference_logps

    def spy(ref_model, get_hidden, get_lm_head, input_ids, attention_mask, labels, num_chunks=1, forward_kwargs=None):
        seen.append({k: tuple(v.shape) for k, v in (forward_kwargs or {}).items()})
        assert type(ref_model).__name__ == "Qwen3_5ForConditionalGeneration"
        return reference_logps(
            ref_model, get_hidden, get_lm_head, input_ids, attention_mask, labels, num_chunks, forward_kwargs
        )

    monkeypatch.setattr("palingenesis.train.reference_logps", spy)
    result = run_cpu(config)
    assert result.global_step == 3
    assert "eval/rewards/accuracies" in result.metrics or result.best_eval_loss is not None
    # the frozen reference read the pixels of both sides of every pair: chosen rows, then rejected
    assert seen and all({"pixel_values", "image_grid_thw", "position_ids"} <= set(s) for s in seen)
    pairs = config.train.per_device_batch_size
    assert all(s["image_grid_thw"][0] == 2 * pairs and s["position_ids"][:2] == (3, 2 * pairs) for s in seen)


def test_preference_rows_carry_images_on_both_sides(tmp_path):
    from multimodal_helpers import tiny_image_processor

    from palingenesis.dpo import PreferenceDataset, collate_preferences
    from palingenesis.multimodal import VisionProcessor

    tok = tiny_tokenizer()
    vision = VisionProcessor(tiny_image_processor(), tok, "qwen3_5")
    rows = [json.loads(line) for line in open(_write_pairs(tmp_path, 2))]
    ds = PreferenceDataset(rows, tok, 256)
    ds._chosen.vision = ds._rejected.vision = vision
    pairs = list(ds)
    assert len(pairs) == 2
    for pair in pairs:
        assert torch.equal(pair["chosen"]["pixel_values"], pair["rejected"]["pixel_values"])
        assert (pair["chosen"]["labels"][pair["chosen"]["input_ids"] == vision.image_token_id] == -100).all()
    batch = collate_preferences(pairs, tok.pad_token_id)
    assert batch["image_grid_thw"].shape[0] == 4 and batch[POSITION_FIELD].shape[0] == 4


def _distill_config(tmp_path, teacher_dir):
    student = save_tiny(tmp_path / "student", seed=0)
    config = cpu_config(tmp_path, student, _write_image_rows(tmp_path), steps=3)
    config.multimodal.enabled = True
    config.data.max_seq_length = 256
    config.distill.enabled = True
    config.distill.teachers = {"t": {"model": str(teacher_dir), "dtype": "float32"}}
    config.distill.ce_weight = 0.5
    return config


def test_online_teacher_reads_the_same_pixels(tmp_path, monkeypatch):
    from palingenesis.distill.teachers import OnlineTeacher

    teacher_dir = save_tiny(tmp_path / "teacher", seed=1, hidden=48)
    config = _distill_config(tmp_path, teacher_dir)
    calls = []
    hidden_states = OnlineTeacher.hidden_states

    def spy(self, input_ids, forward_kwargs, compute_dtype, autocast):
        calls.append((input_ids.clone(), {k: v.clone() for k, v in forward_kwargs.items() if torch.is_tensor(v)}))
        return hidden_states(self, input_ids, forward_kwargs, compute_dtype, autocast)

    monkeypatch.setattr(OnlineTeacher, "hidden_states", spy)
    result = run_cpu(config)
    assert result.global_step == 3
    assert result.metrics["train/kd_coverage"] == 1.0 and result.metrics["train/kd_loss"] > 0
    ids, kwargs = calls[0]
    assert {"pixel_values", "image_grid_thw", "position_ids"} <= set(kwargs)
    image_tokens = int((ids == tiny_tokenizer().convert_tokens_to_ids("<|image_pad|>")).sum())
    assert image_tokens == int((kwargs["image_grid_thw"].prod(-1) // 4).sum()) > 0


def test_teacher_targets_depend_on_the_pixels(tmp_path):
    """The teacher's hidden states at the trained positions change with the image: it is not text-only."""
    from multimodal_helpers import tiny_image_processor, tiny_qwen3_5

    from palingenesis.data import ChatDataset, _collate_fn
    from palingenesis.logits import final_hidden_states
    from palingenesis.multimodal import VisionProcessor, forward_kwargs

    tok = tiny_tokenizer()
    vision = VisionProcessor(tiny_image_processor(), tok, "qwen3_5")
    teacher = tiny_qwen3_5(tok, hidden=48, seed=1).eval()

    def hidden(seed):
        row = {
            "messages": [
                {"role": "user", "content": [{"type": "image", "image": tiny_image(24, 24, seed)}]},
                {"role": "assistant", "content": "A picture."},
            ]
        }
        batch = _collate_fn(list(ChatDataset([row], tok, 128, vision=vision)), tok.pad_token_id)
        with torch.no_grad():
            return final_hidden_states(teacher, batch["input_ids"], None, **forward_kwargs(batch, {}))

    assert not torch.allclose(hidden(0), hidden(1), atol=1e-4)
    assert torch.allclose(hidden(0), hidden(0))


def test_text_only_teacher_is_refused(tmp_path):
    save_model(tmp_path / "teacher", tiny_tokenizer(), seed=1)
    config = _distill_config(tmp_path, tmp_path / "teacher")
    with pytest.raises(MultimodalError, match="not a vision-language checkpoint"):
        run_cpu(config)


def test_cached_teacher_is_refused(tmp_path):
    teacher_dir = save_tiny(tmp_path / "teacher", seed=1)
    config = _distill_config(tmp_path, teacher_dir)
    config.distill.teachers = {"t": {"cache": str(tmp_path / "cache")}}
    with pytest.raises(ConfigError, match="cached teacher"):
        run_cpu(config)
