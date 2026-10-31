"""`pgs internals run`: capture every model on every dataset, sweep probes per model, compare model pairs.

Output layout (under output_dir):

    acts/<model>/<dataset>/        activation stores (reused when present and captured with the same settings)
    probes/<model>/                layer_<r>.safetensors, best.safetensors, sweep.json
    compare/<a>__<b>.json          per-layer CKA, cosine, probe transfer A -> B
    report.json                    everything above in one file, plus the config
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import torch

from palingenesis.internals.capture import ActivationStore, capture_dataset, load_rows, model_fingerprint
from palingenesis.internals.config import InternalsConfig, InternalsConfigError
from palingenesis.internals.probes import sweep
from palingenesis.internals.similarity import compare_stores

logger = logging.getLogger(__name__)
_TORCH_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(path: str, dtype: str | None = None, device: str | None = None):
    """(model, tokenizer) in eval mode; dtype default bf16 on CUDA, fp32 on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dt = _TORCH_DTYPES[dtype] if dtype else (torch.bfloat16 if dev.type == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dt, low_cpu_mem_usage=True).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def _reusable(path: Path, settings: dict, weights_sha: str) -> bool:
    try:
        store = ActivationStore(path)
    except (FileNotFoundError, ValueError):
        return False
    meta = store.meta
    return meta["model"].get("weights_sha") == weights_sha and all(meta.get(k) == v for k, v in settings.items())


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_internals(config: InternalsConfig) -> dict:
    errors = config.validate()
    if errors:
        raise InternalsConfigError("; ".join(errors))
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    datasets = {"train": config.train, **config.tests}
    rows = {name: load_rows(path, config.limit) for name, path in datasets.items()}
    stores: dict[str, dict[str, ActivationStore]] = {}
    for name, path in config.models.items():
        model, tok = load_model(path, config.model_dtype, config.device)
        sha = model_fingerprint(model, tok)["weights_sha"]
        stores[name] = {}
        for ds, ds_rows in rows.items():
            where = out / "acts" / name / ds
            settings = {
                "position": config.position,
                "source": str(datasets[ds]),
                "source_rows": len(ds_rows),
                "max_tokens": config.max_tokens,
                "chat_template_kwargs": config.chat_template_kwargs,
                "dtype": config.store_dtype,
                "layers_spec": config.layers,
            }
            if not config.overwrite and _reusable(where, settings, sha):
                logger.info("internals run: reusing %s", where)
                stores[name][ds] = ActivationStore(where)
                continue
            stores[name][ds] = capture_dataset(
                model,
                tok,
                ds_rows,
                where,
                layers=config.layers,
                position=config.position,
                chat_template_kwargs=config.chat_template_kwargs,
                batch_size=config.batch_size,
                max_batch_tokens=config.max_batch_tokens,
                max_tokens=config.max_tokens,
                dtype=config.store_dtype,
                overwrite=True,
                extra_meta={"source": str(datasets[ds]), "max_tokens": config.max_tokens, "layers_spec": config.layers},
            )
        del model
        _free()

    report: dict = {"config": config.to_dict(), "sweeps": {}, "compare": {}}
    for name in config.models:
        tests = {ds: s for ds, s in stores[name].items() if ds != "train"}
        res = sweep(
            stores[name]["train"],
            kind=config.probe_kind,
            tests=tests,
            val=config.val,
            test=config.test,
            seed=config.seed,
            resamples=config.bootstrap,
            label_threshold=config.label_threshold,
            control=config.control,
            save_dir=out / "probes" / name,
        )
        res.pop("probes")
        report["sweeps"][name] = res
    for a, b in config.compare:
        res = compare_stores(
            stores[a]["train"],
            stores[b]["train"],
            probe_kind=config.probe_kind,
            val=config.val,
            test=config.test,
            seed=config.seed,
            resamples=config.bootstrap,
            label_threshold=config.label_threshold,
        )
        (out / "compare").mkdir(exist_ok=True)
        (out / "compare" / f"{a}__{b}.json").write_text(json.dumps(res, indent=2))
        report["compare"][f"{a}__{b}"] = res
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report
