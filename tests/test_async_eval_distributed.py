"""eval_tasks async push under FSDP2 (pgs train): two CPU processes (gloo). Rank 0 decides whether a due step
pushes and every rank learns it (the snapshot gathers, a collective); the evaluator receives the full weights."""

import asyncio
import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _tiny():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=96,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )  # fmt: skip
    )


class _Capture:
    """An evaluator that keeps the first push it receives and never acknowledges a second one."""

    def __init__(self, run: Path):
        self.received = []
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        port = asyncio.run_coroutine_threadsafe(self._start(), self.loop).result()
        from palingenesis.eval import handoff

        handoff.write_json(handoff.root(run) / handoff.SERVICE, {"mode": "serve", "state": "listening",
                                                                 "host": "127.0.0.1", "port": port})  # fmt: skip

    async def _start(self):
        server = await asyncio.start_server(self._connection, "127.0.0.1", 0)
        return server.sockets[0].getsockname()[1]

    async def _connection(self, reader, writer):
        from palingenesis.rl.pool import buckets as bucketed
        from palingenesis.rl.pool.protocol import receive, send

        await receive(reader)
        await send(writer, {"op": "hello", "ok": True})
        while True:
            header, tensors = await receive(reader)
            if header.get("op") != "weights":
                continue
            spec = header["buckets"]
            buffers = [tensors[f"bucket/{i}"] for i in range(len(spec["sizes"]))]
            assert bucketed.verify(spec, buffers) == ""
            self.received.append((header["step"], {k: v.clone() for k, v in bucketed.unpack(spec, buffers).items()}))
            if len(self.received) == 1:
                await send(writer, {"op": "received", "step": header["step"]})


def _worker(rank: int, world: int, init_file: str, out_dir: str):
    import palingenesis.eval.push as push
    from palingenesis.eval.async_eval import AsyncEval, RunInfo, broadcast_flag, full_weights
    from palingenesis.eval.config import EvalTasksConfig

    push.RECONNECT_SECONDS = 0.05
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    model = _tiny()
    reference = {k: v.clone() for k, v in model.state_dict().items()}
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    run = Path(out_dir) / "run"
    task = {"name": "t", "dataset": "rows.jsonl", "prompt": "{{q}}", "scorer": "numeric"}
    spec = EvalTasksConfig(tasks=[task], placement="async", service="auto", every=2, final_timeout_s=0)
    info = RunInfo(trainer="train", model="m", output_dir=str(run), every=2, total_steps=10)
    capture = None
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        capture = _Capture(run)
    dist.barrier()
    evaluator = AsyncEval(spec, info, main=rank == 0, agree=lambda flag: broadcast_flag(flag, "cpu"))
    evaluator.start()
    if rank == 0:
        import time

        deadline = time.monotonic() + 30
        while not evaluator.client.ready():
            assert time.monotonic() < deadline
            time.sleep(0.05)
    dist.barrier()
    decisions = []
    for step in (2, 4, 6):
        wanted = evaluator.wants_push(step, 10)
        decisions.append(wanted)
        if wanted:
            evaluator.push(step, full_weights(model, fsdp=True, main=rank == 0))
        if rank == 0 and step == 2:
            import time

            deadline = time.monotonic() + 30
            while evaluator.client.sending:
                assert time.monotonic() < deadline
                time.sleep(0.05)
    result = {"decisions": decisions}
    if rank == 0:
        import time

        deadline = time.monotonic() + 30
        while len(capture.received) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        first_step, weights = capture.received[0]
        result["steps"] = [s for s, _ in capture.received]
        result["names"] = sorted(weights)
        result["expected"] = sorted(k for k in reference if k != "lm_head.weight")
        result["equal"] = all(torch.equal(weights[k], reference[k].to(torch.bfloat16)) for k in weights)
        result["skipped"] = evaluator.counts["push_skipped"]
        (Path(out_dir) / "result.json").write_text(json.dumps(result))
    evaluator.close(wait=False)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed")
def test_fsdp_ranks_agree_and_the_evaluator_gets_the_full_weights():
    with tempfile.TemporaryDirectory() as tmp:
        init = os.path.join(tmp, "init")
        mp.spawn(_worker, args=(2, init, tmp), nprocs=2, join=True)
        result = json.loads(Path(tmp, "result.json").read_text())
    # step 2 pushed and was acknowledged; step 4 pushed and never acknowledged; step 6 skipped on every rank
    assert result["decisions"] == [True, True, False]
    assert result["steps"] == [2, 4] and result["skipped"] == 1
    assert result["names"] == result["expected"]  # the full state, under checkpoint names, no tied copy
    assert result["equal"]
