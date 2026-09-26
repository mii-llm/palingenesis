"""GSM8K as RL rows: {"prompt", "answer"} with the final number as the answer.

python examples/rl/prepare_gsm8k.py            # -> data/gsm8k_train.jsonl, data/gsm8k_test.jsonl
"""

import json
from pathlib import Path

from datasets import load_dataset

out = Path("data")
out.mkdir(exist_ok=True)
for split in ("train", "test"):
    rows = load_dataset("openai/gsm8k", "main", split=split)
    with open(out / f"gsm8k_{split}.jsonl", "w") as f:
        for row in rows:
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            f.write(json.dumps({"prompt": row["question"], "answer": answer}) + "\n")
    print(f"{out / f'gsm8k_{split}.jsonl'}: {len(rows)} rows")
