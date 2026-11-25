"""Monthly output per GPU from SemiAnalysis InferenceX benchmark history.

InferenceX (formerly InferenceMAX, open source under Apache 2.0) runs serving benchmarks on
many chips. Its dashboard reads a public JSON feed; this module reads the same feed.

For each chip and day we keep the best output tokens/s per GPU that meets a speed target
(median tokens/s per user >= min_user_tps and p99 time to first token <= max_ttft_s), plus the
best with no target ("batch"). Days are reduced to a monthly median. Only single-node runs on
the chosen engine are used, so chips are compared on the same software.
"""

from __future__ import annotations

import collections
import statistics
from urllib.parse import urlencode

from . import fetch_json

HISTORY_URL = "https://inferencex.semianalysis.com/api/v1/benchmarks/history?"


def fetch_history(model: str, isl: int = 1024, osl: int = 1024, fetch=fetch_json) -> list[dict]:
    return fetch(HISTORY_URL + urlencode({"model": model, "isl": isl, "osl": osl}), timeout=300)


def monthly_output(rows: list[dict], chips: tuple[str, ...] = ("h100", "h200", "b200"),
                   engine: str = "vllm", min_user_tps: float = 100.0, max_ttft_s: float = 2.0,
                   precision: str | None = None) -> list[dict]:
    rows = [r for r in rows
            if r.get("benchmark_type") == "single_turn" and not r.get("is_multinode")
            and r.get("hardware") in chips and r.get("framework") == engine
            and (precision is None or r.get("precision") == precision)]

    shares = [r["metrics"]["output_tput_per_gpu"] / r["metrics"]["tput_per_gpu"]
              for r in rows if r["metrics"].get("output_tput_per_gpu") and r["metrics"].get("tput_per_gpu")]
    output_share = statistics.median(shares) if shares else 0.5

    def out_tps(m: dict) -> float:
        return m.get("output_tput_per_gpu") or m["tput_per_gpu"] * output_share

    daily = collections.defaultdict(lambda: {"interactive": 0.0, "batch": 0.0, "images": set()})
    for r in rows:
        m = r["metrics"]
        day = daily[(r["hardware"], r["date"])]
        value = out_tps(m)
        day["batch"] = max(day["batch"], value)
        if m.get("median_intvty", 0) >= min_user_tps and m.get("p99_ttft", 1e9) <= max_ttft_s:
            day["interactive"] = max(day["interactive"], value)
        if r.get("image"):
            day["images"].add(r["image"])

    months = collections.defaultdict(lambda: {"interactive": [], "batch": [], "images": set(), "days": 0})
    for (chip, date), day in daily.items():
        month = months[(chip, date[:7])]
        month["days"] += 1
        month["images"] |= day["images"]
        for key in ("interactive", "batch"):
            if day[key] > 0:
                month[key].append(day[key])

    out = []
    for (chip, month), v in sorted(months.items()):
        if not v["interactive"] and not v["batch"]:
            continue
        out.append({
            "chip": chip, "month": month, "days": v["days"],
            "interactive_out_tps": round(statistics.median(v["interactive"]), 1) if v["interactive"] else None,
            "batch_out_tps": round(statistics.median(v["batch"]), 1) if v["batch"] else None,
            "engine_images": sorted(v["images"]),
        })
    return out
