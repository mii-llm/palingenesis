"""Quantised checkpoints in the compressed-tensors format, which vLLM loads (`quant_method: compressed-tensors`).

Written without the compressed-tensors package, from its specification (vllm-project/compressed-tensors, Apache-2.0)
and vLLM's loaders (vllm/model_executor/layers/quantization/compressed_tensors):

  scheme              format                  tensors per linear layer
  nvfp4 (+ none)      nvfp4-pack-quantized    weight_packed u8 [out, in/2], weight_scale f8e4m3 [out, in/16],
                                              weight_global_scale f32 [1] (= 448 * 6 / amax, a divisor)
  nvfp4 + nvfp4                               + input_global_scale f32 [1] (static, calibrated; block scales dynamic)
  mxfp4 (+ mxfp4)     mxfp4-pack-quantized    weight_packed u8 [out, in/2], weight_scale u8 [out, in/32] (E8M0)
  fp8 (+ fp8)         float-quantized         weight f8e4m3 [out, in], weight_scale f32 [out, 1]
  fp8_block (+ fp8)   float-quantized         weight f8e4m3 [out, in], weight_scale f32 [out/128, in/128]

E2M1 codes are packed two per byte, the even input column in the low nibble, sign in bit 3. A linear layer whose
weights stay in full precision (excluded, a shape the format does not take, the head) is listed in the config's
`ignore`; the embedding and head keep their (fake-quantised, if trained so) values in bf16.

Why this format: it is the one vLLM loads for every format here (ModelOpt's covers NVFP4 and FP8 but has no dense
MXFP4), and it runs on GPUs without FP4 / FP8 tensor cores through vLLM's Marlin weight-only kernels (NVFP4 and
MXFP4 from compute capability 7.5 / 8.0, FP8 from 7.5): an A100 serves these checkpoints with 4-bit / 8-bit weights
and bf16 activations; the activation formats need Blackwell (FP4) or Hopper / Ada (FP8) to run as trained.

`pgs quantize` (main below) is post-training quantisation with the same code: round-to-nearest with max or
MSE-optimal scales, written in both forms (packed for vLLM, dequantized bf16 for any loader).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
from torch import Tensor

from palingenesis.qat.formats import Quantized, e8m0_encode, global_scale_from_amax, pack_e2m1

logger = logging.getLogger(__name__)

_FP4_WEIGHTS = {"num_bits": 4, "type": "float", "symmetric": True, "dynamic": False}
_FP8 = {"num_bits": 8, "type": "float", "symmetric": True}
# (weight format, activations) -> (compressed-tensors format, weights args, input activation args)
SCHEMES: dict[tuple[str, str], tuple[str, dict, dict | None]] = {
    ("nvfp4", "none"): (
        "nvfp4-pack-quantized",
        {**_FP4_WEIGHTS, "strategy": "tensor_group", "group_size": 16, "scale_dtype": "torch.float8_e4m3fn"},
        None,
    ),
    ("nvfp4", "nvfp4"): (
        "nvfp4-pack-quantized",
        {**_FP4_WEIGHTS, "strategy": "tensor_group", "group_size": 16, "scale_dtype": "torch.float8_e4m3fn"},
        {
            "num_bits": 4,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor_group",
            "group_size": 16,
            "dynamic": "local",
            "scale_dtype": "torch.float8_e4m3fn",
        },
    ),
    ("mxfp4", "none"): (
        "mxfp4-pack-quantized",
        {**_FP4_WEIGHTS, "strategy": "group", "group_size": 32, "scale_dtype": "torch.uint8"},
        None,
    ),
    ("mxfp4", "mxfp4"): (
        "mxfp4-pack-quantized",
        {**_FP4_WEIGHTS, "strategy": "group", "group_size": 32, "scale_dtype": "torch.uint8"},
        {
            "num_bits": 4,
            "type": "float",
            "symmetric": True,
            "strategy": "group",
            "group_size": 32,
            "dynamic": True,
            "scale_dtype": "torch.uint8",
        },
    ),
    ("fp8", "none"): ("float-quantized", {**_FP8, "strategy": "channel", "dynamic": False}, None),
    ("fp8", "fp8"): (
        "float-quantized",
        {**_FP8, "strategy": "channel", "dynamic": False},
        {**_FP8, "strategy": "token", "dynamic": True},
    ),
    ("fp8_block", "none"): (
        "float-quantized",
        {**_FP8, "strategy": "block", "block_structure": [128, 128], "dynamic": False},
        None,
    ),
    ("fp8_block", "fp8"): (
        "float-quantized",
        {**_FP8, "strategy": "block", "block_structure": [128, 128], "dynamic": False},
        {**_FP8, "strategy": "group", "group_size": 128, "dynamic": True},
    ),
}


def scheme_for(weight_format: str, activations: str) -> tuple[str, str] | None:
    """The SCHEMES key of a format pair, or None when vLLM has no checkpoint format for it."""
    key = (weight_format, activations)
    return key if key in SCHEMES else None


def pack(q: Quantized, activation_amax: Tensor | None = None) -> dict[str, Tensor]:
    """A quantised weight as the compressed-tensors tensors of its layer ({suffix: tensor}, on the CPU)."""
    name = q.spec.name
    out: dict[str, Tensor] = {}
    if name in ("nvfp4", "mxfp4"):
        out["weight_packed"] = pack_e2m1(q.values).cpu()
        if name == "nvfp4":
            out["weight_scale"] = q.scale.to(torch.float8_e4m3fn).cpu()
            out["weight_global_scale"] = q.global_scale.reshape(1).to(torch.float32).cpu()
        else:
            out["weight_scale"] = e8m0_encode(q.scale).cpu()
    else:
        out["weight"] = q.values.to(torch.float8_e4m3fn).cpu()
        out["weight_scale"] = q.scale.to(torch.float32).cpu()
    if activation_amax is not None:
        out["input_global_scale"] = global_scale_from_amax(activation_amax.float()).reshape(1).cpu()
    return out


def unpack(tensors: dict[str, Tensor], scheme: tuple[str, str]) -> Tensor:
    """The float32 weight a packed layer holds, computed as vLLM's loaders define it (a reference for tests and
    for checking an export): nvfp4 q * (weight_scale / weight_global_scale), mxfp4 q * 2^e, fp8 q * scale."""
    from palingenesis.qat.formats import e8m0_decode, unpack_e2m1

    fmt = scheme[0]
    if fmt in ("nvfp4", "mxfp4"):
        values = unpack_e2m1(tensors["weight_packed"])
        if fmt == "nvfp4":
            scale = tensors["weight_scale"].float() / tensors["weight_global_scale"].float()
            block = 16
        else:
            scale = e8m0_decode(tensors["weight_scale"])
            block = 32
        rows, cols = values.shape
        return (values.reshape(rows, cols // block, block) * scale.unsqueeze(-1)).reshape(rows, cols)
    values = tensors["weight"].float()
    scale = tensors["weight_scale"].float()
    if fmt == "fp8":
        return values * scale
    rows, cols = values.shape
    return (values.reshape(rows // 128, 128, cols // 128, 128) * scale[:, None, :, None]).reshape(rows, cols)


def quantization_config(scheme: tuple[str, str], ignore: list[str]) -> dict:
    """config.json's quantization_config for a scheme (every quantised linear layer in one group)."""
    fmt, weights, inputs = SCHEMES[scheme]
    return {
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": dict(weights),
                "input_activations": dict(inputs) if inputs else None,
                "output_activations": None,
                "format": fmt,
            }
        },
        "format": fmt,
        "global_compression_ratio": None,
        "ignore": sorted(ignore),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
    }


def _module_tail(key: str) -> str:
    return key.removeprefix("model.")


def write_checkpoint(
    source: Path,
    target: Path,
    pending: dict[str, tuple[Tensor, dict[str, Tensor]]],
    scheme: tuple[str, str],
    report: dict | None = None,
) -> None:
    """Write the compressed-tensors checkpoint `target` from the exported model at `source`.

    pending: {state key of a quantised weight: (the tensor written to `source` for it, its packed tensors)}.
    The source's files are read as saved (names in the checkpoint's own layout, e.g. Qwen3.5's multimodal one):
    each quantised weight is found by name (its module path, as a dotted suffix) and verified by value, then
    replaced by its packed tensors. Every other tensor is copied (float32 ones as bf16), every non-weight file
    (config, tokenizer, chat template, processors) too; config.json gets the quantization_config."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, target = Path(source), Path(target)
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"{source}: no *.safetensors to quantise")
    if target.exists() and any(target.iterdir()):
        for old in sorted(target.glob("*.safetensors")) + sorted(target.glob("*.json")):
            old.unlink()  # a previous export of the same name (best/ is replaced at every improvement)
    target.mkdir(parents=True, exist_ok=True)

    by_tail: dict[str, list[str]] = {}
    for key in pending:
        by_tail.setdefault(_module_tail(key), []).append(key)
    matched: dict[str, str] = {}  # pending key -> saved key
    ignore: list[str] = []
    weight_map: dict[str, str] = {}
    total = 0
    for file in files:
        out: dict[str, Tensor] = {}
        with safe_open(str(file), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                found = _match(key, tensor, by_tail, pending, matched)
                if found is not None:
                    matched[found] = key
                    prefix = key[: -len("weight")]
                    for suffix, packed in pending[found][1].items():
                        out[prefix + suffix] = packed.contiguous()
                    continue
                if key.endswith(".weight") and tensor.dim() >= 2:
                    ignore.append(key[: -len(".weight")])  # a linear layer (or embedding) left in full precision
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16)
                out[key] = tensor.contiguous()
        save_file(out, str(target / file.name), metadata={"format": "pt"})
        for key, tensor in out.items():
            weight_map[key] = file.name
            total += tensor.numel() * tensor.element_size()
    missing = sorted(set(pending) - set(matched))
    if missing:
        raise RuntimeError(
            f"quantised export: {len(missing)} quantised weight(s) not found in {source} under their module names "
            f"with their exported values ({missing[:3]}): the checkpoint layout renames or transforms them"
        )
    if len(files) > 1:
        index = {"metadata": {"total_size": total}, "weight_map": weight_map}
        (target / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    for extra in sorted(source.iterdir()):
        if extra.is_file() and extra.suffix != ".safetensors" and extra.name != "model.safetensors.index.json":
            shutil.copy(extra, target / extra.name)
    if "lm_head" not in ignore and not any(k.endswith(".lm_head") for k in ignore):
        ignore.append("lm_head")  # vLLM never quantises the head here; listed as llm-compressor does
    config_file = target / "config.json"
    config = json.loads(config_file.read_text())
    config["quantization_config"] = quantization_config(scheme, ignore)
    for holder in (config, config.get("text_config") or {}):
        for k in ("dtype", "torch_dtype"):
            if holder.get(k) == "float32":
                holder[k] = "bfloat16"
    config_file.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if report:
        (target / "quantization_report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


def _match(key: str, tensor: Tensor, by_tail: dict, pending: dict, matched: dict) -> str | None:
    """The pending key the saved tensor `key` holds: same module path (dotted suffix) and the same values."""
    parts = key.split(".")
    for start in range(len(parts)):
        candidates = by_tail.get(".".join(parts[start:]))
        if not candidates:
            continue
        for cand in candidates:
            if cand in matched:
                continue
            expected = pending[cand][0]
            if expected.shape == tensor.shape and torch.equal(expected.to(tensor.dtype).cpu(), tensor):
                return cand
    return None


# ─────────────────────────────────────────────────────────────── pgs quantize


def main(argv: list[str] | None = None) -> int:
    """pgs quantize: post-training quantisation (round to nearest, max or MSE-optimal scales) of a checkpoint
    into the compressed-tensors format vLLM loads, and optionally its dequantized bf16 twin."""
    import argparse
    import tempfile

    from palingenesis.qat.formats import FORMATS
    from palingenesis.qat.fp import ACTIVATIONS, FloatQAT

    parser = argparse.ArgumentParser(
        prog="pgs quantize",
        description="Post-training FP4 / FP8 quantisation into a vLLM-loadable (compressed-tensors) checkpoint.",
    )
    parser.add_argument("--model", required=True, help="checkpoint directory or Hub id")
    parser.add_argument("--format", required=True, choices=list(FORMATS))
    parser.add_argument("--activations", default="none", choices=list(ACTIVATIONS))
    parser.add_argument("--scales", default="max", choices=["max", "mse"])
    parser.add_argument("--mx_rounding", default="even", choices=["floor", "even", "ceil"])
    parser.add_argument("--exclude", nargs="*", default=None, help="regular expressions of layers to keep")
    parser.add_argument("--out", required=True, help="the packed checkpoint (compressed-tensors)")
    parser.add_argument("--dequantized", default="", help="also write the dequantized bf16 model here")
    parser.add_argument(
        "--calibration", default="", help="NVFP4 activations: a text file whose lines calibrate the input scales"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from palingenesis.checkpoint import save_hf_model
    from palingenesis.hooks import PluginContext

    if args.activations == "nvfp4" and not args.calibration:
        parser.error("--activations nvfp4 needs --calibration (a text file): its input scales are static")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    plugin = FloatQAT(
        format=args.format,
        activations=args.activations,
        scales=args.scales,
        mx_rounding=args.mx_rounding,
        exclude=args.exclude,
        quantized_export=True,
    )
    ctx = PluginContext(config=None, device=torch.device(args.device))
    model = plugin.transform_model(model, ctx)
    plugin.on_train_start(model, ctx)
    if args.calibration:
        model.train()  # the activation observers record in training mode
        lines = [ln for ln in Path(args.calibration).read_text().splitlines() if ln.strip()]
        with torch.no_grad():
            for line in lines:
                ids = tok(line, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(args.device)
                model(input_ids=ids)
        model.eval()
    state = plugin.export_state_dict(dict(model.state_dict()), model, ctx)
    out = Path(args.out)
    out.resolve().parent.mkdir(parents=True, exist_ok=True)
    if args.dequantized:
        save_hf_model(model, tok, Path(args.dequantized), state=state, source_layout=True)
        write_checkpoint(Path(args.dequantized), out, plugin._pending, plugin.scheme, plugin._report)
    else:  # the dequantized model only as the packed one's source (same layout and files), then removed
        with tempfile.TemporaryDirectory(dir=out.resolve().parent) as tmp:
            save_hf_model(model, tok, Path(tmp) / "dequantized", state=state, source_layout=True)
            write_checkpoint(Path(tmp) / "dequantized", out, plugin._pending, plugin.scheme, plugin._report)
    errors = [r["relative_error"] for r in plugin._report.values()]
    print(
        json.dumps(
            {
                "out": str(out),
                "format": args.format,
                "activations": args.activations,
                "layers": len(errors),
                "relative_error_mean": sum(errors) / len(errors),
                "relative_error_max": max(errors),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
