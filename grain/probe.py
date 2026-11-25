"""Run one probe slot: the same tasks per model, sent to each pinned provider.

Results go to data/probes/YYYY/MM/YYYY-MM-DD_HH.jsonl, one row per task. Prompts are not
stored (private tasks must stay private); the task id and type are.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .grade import grade
from .openrouter import chat_stream
from .tasks import PUBLIC_SEED, make_task


def task_indices(day: dt.date, slot_hour: int, n: int) -> list[int]:
    """Fresh indices for every day and slot, so no private task is ever sent twice."""
    base = (day - dt.date(2026, 1, 1)).days * 10_000 + slot_hour * 100
    return [base + k for k in range(n)]


def run_slot(models_cfg: dict, probe_cfg: dict, api_key: str, seed: str | None,
             day: dt.date, slot_hour: int, out_dir: Path, client=chat_stream,
             log=print) -> Path:
    label = "private" if seed else "public"
    seed = seed or PUBLIC_SEED
    if label == "public":
        log("warning: GRAIN_PRIVATE_SEED is not set, probing with the public task suite")

    n = probe_cfg["tasks_per_slot"]
    indices = task_indices(day, slot_hour, n)
    jobs_by_provider: dict[str, list] = defaultdict(list)
    for model in models_cfg["models"]:
        tasks = [make_task(seed, i, probe_cfg["type_cycle"], probe_cfg["target_input_tokens"], label)
                 for i in indices]
        for provider in model["providers"]:
            for task in tasks:
                jobs_by_provider[provider].append((model, task))

    budget = probe_cfg["max_usd_per_run"]
    spent = [0.0]
    lock = threading.Lock()
    rows: list[dict] = []
    speed_floor = probe_cfg["interactive"]["min_tokens_per_s"]
    max_ttft = probe_cfg["interactive"]["max_ttft_s"]

    def work(provider: str, jobs: list) -> None:
        for model, task in jobs:
            with lock:
                over = spent[0] >= budget
            base = {
                "ts_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "date": day.isoformat(), "slot": slot_hour, "suite": label,
                "model_code": model["code"], "model": model["openrouter"], "provider": provider,
                "task_id": task.id, "task_type": task.type,
            }
            if over:
                rows.append({**base, "skipped": "budget"})
                continue
            result = client(api_key, model["openrouter"], provider, task.prompt,
                            probe_cfg["max_tokens"][task.type], model.get("reasoning"),
                            probe_cfg["timeout_s"])
            passed = result["error"] is None and grade(task, result["text"])
            meets = (passed and result["ttft_s"] is not None and result["ttft_s"] <= max_ttft
                     and (result["tokens_per_s"] or 0) >= speed_floor[model["code"]])
            text = result.pop("text")
            row = {**base, **result, "passed": passed, "meets_interactive": meets,
                   "answer_excerpt": text[-160:]}
            with lock:
                spent[0] += result["cost_usd"] or 0.0
                rows.append(row)

    with ThreadPoolExecutor(max_workers=max(1, len(jobs_by_provider))) as pool:
        for provider, jobs in jobs_by_provider.items():
            pool.submit(work, provider, jobs)

    out = out_dir / "probes" / f"{day:%Y}" / f"{day:%m}" / f"{day:%Y-%m-%d}_{slot_hour:02d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["model_code"], r["provider"], r["task_id"]))
    with out.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    done = [r for r in rows if "skipped" not in r]
    log(f"{len(done)} tasks sent, {sum(r['passed'] for r in done)} passed, "
        f"{sum(1 for r in rows if 'skipped' in r)} skipped, ${spent[0]:.4f} spent -> {out}")
    return out


def nearest_slot(hour: int, slots: list[int]) -> int:
    return min(slots, key=lambda s: min(abs(hour - s), 24 - abs(hour - s)))


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())
