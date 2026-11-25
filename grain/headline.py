"""Headline numbers: exchange rate, break-even utilization and software gain, per chip and month.

Output per GPU comes from InferenceX (1,024 input / 1,024 output token requests, so one request
stands in for one Grain until the task suite's own token counts replace it). Revenue uses the
median list price across the probed providers; rent uses GetDeploying monthly medians.
After the last InferenceX month, output is held flat ("held": true) while rent keeps updating.
"""

from __future__ import annotations

import collections
import csv
import json
import math
import statistics
from pathlib import Path

from .sources.prices import median_list_price

BASE_CHIP = "h100"
INPUT_FILL = 0.9  # InferenceX draws each request's input at 80-100% of the target length
MAX_HOLD_MONTHS = 6  # carry the last measured output forward at most this long


def revenue_per_gpu_hour(out_tps: float, in_per_m: float, out_per_m: float,
                         isl: int = 1024, osl: int = 1024) -> float:
    requests_per_hour = out_tps * 3600 / osl
    usd_per_request = (INPUT_FILL * isl * in_per_m + osl * out_per_m) / 1e6
    return requests_per_hour * usd_per_request


def _add_months(month: str, n: int) -> str:
    total = int(month[:4]) * 12 + int(month[5:7]) - 1 + n
    return f"{total // 12}-{total % 12 + 1:02d}"


def _month_range(first: str, last: str) -> list[str]:
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y}-{m:02d}" <= last:
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def price_for_month(snapshots: dict[str, dict], month: str, model: dict) -> tuple[float, float, str]:
    """List price from that month's first snapshot, else the nearest later one, else the fallback."""
    candidates = sorted(d for d in snapshots if d[:7] >= month) or sorted(snapshots)
    for day in candidates:
        found = median_list_price(snapshots[day], model["code"], model["providers"])
        if found:
            return found[0], found[1], f"openrouter {day}"
    fb = model["list_price_fallback"]
    return fb["in_per_m"], fb["out_per_m"], "config fallback"


def compute(models_cfg: dict, inferencex: dict[str, list[dict]], rent: list[dict],
            snapshots: dict[str, dict]) -> list[dict]:
    rent_by = {(r["chip"], r["month"]): r["usd_per_gpu_hour"] for r in rent}
    last_rent_month = max(r["month"] for r in rent) if rent else None
    rows = []
    for model in models_cfg["models"]:
        series = inferencex.get(model["code"])
        if not series:
            continue
        by_chip = collections.defaultdict(dict)
        for r in series:
            if r["interactive_out_tps"]:
                by_chip[r["chip"]][r["month"]] = r["interactive_out_tps"]
        base_series = by_chip.get(BASE_CHIP, {})
        base_first = base_series[min(base_series)] if base_series else None
        for chip, measured in by_chip.items():
            last_measured = max(measured)
            hold_limit = _add_months(last_measured, MAX_HOLD_MONTHS)
            end = max(last_measured, min(last_rent_month or last_measured, hold_limit))
            for month in _month_range(min(measured), end):
                held = month > last_measured
                if month in measured:
                    out_tps = measured[month]
                elif held:
                    out_tps = measured[last_measured]
                else:
                    continue  # a gap inside the measured period stays a gap
                row = {"model": model["code"], "chip": chip, "month": month, "held": held,
                       "interactive_out_tps": out_tps}
                if not held and month in base_series:
                    row["exchange_rate"] = round(out_tps / base_series[month], 2)
                if chip == BASE_CHIP and not held and base_first:
                    row["software_gain_index"] = round(100 * out_tps / base_first, 1)
                in_p, out_p, price_src = price_for_month(snapshots, month, model)
                revenue = revenue_per_gpu_hour(out_tps, in_p, out_p)
                row.update({"revenue_usd_per_gpu_hour": round(revenue, 2), "price_source": price_src})
                rent_usd = rent_by.get((chip, month))
                if rent_usd:
                    row.update({"rent_usd_per_gpu_hour": rent_usd,
                                "breakeven_utilization_pct": round(100 * rent_usd / revenue),
                                "spread_usd_per_gpu_hour": round(revenue - rent_usd, 2)})
                rows.append(row)
    rows.sort(key=lambda r: (r["model"], r["chip"], r["month"]))
    return rows


def composite_exchange_rates(rows: list[dict]) -> list[dict]:
    """Geometric mean across models of each chip's exchange rate, per month."""
    grouped = collections.defaultdict(list)
    for r in rows:
        if "exchange_rate" in r and r["chip"] != BASE_CHIP:
            grouped[(r["chip"], r["month"])].append(r["exchange_rate"])
    return [{"chip": chip, "month": month, "models": len(v),
             "composite_exchange_rate": round(math.exp(statistics.fmean(math.log(x) for x in v)), 2)}
            for (chip, month), v in sorted(grouped.items())]


def field_summary(probe_rows: list[dict]) -> list[dict]:
    """Table D: price per Grain and service quality per provider and model."""
    groups = collections.defaultdict(list)
    for r in probe_rows:
        if "skipped" not in r:
            groups[(r["model_code"], r["provider"])].append(r)
    out = []
    for (model, provider), rs in sorted(groups.items()):
        passed = [r for r in rs if r["passed"]]
        cost = sum(r.get("cost_usd") or 0 for r in rs)
        errors = [r for r in rs if r.get("error")]
        out.append({
            "model": model, "provider": provider, "tasks": len(rs),
            "pass_rate": round(len(passed) / len(rs), 3),
            "usd_per_grain": round(cost / len(passed), 6) if passed else None,
            "interactive_share": round(sum(r["meets_interactive"] for r in rs) / len(rs), 3),
            "failure_rate": round(len(errors) / len(rs), 3),
            "rate_limited": sum(1 for r in errors if r["error"].get("code") == 429),
            "median_ttft_s": statistics.median([r["ttft_s"] for r in rs if r.get("ttft_s")] or [0]) or None,
            "median_tokens_per_s": statistics.median([r["tokens_per_s"] for r in rs if r.get("tokens_per_s")] or [0]) or None,
            "median_out_tokens": statistics.median([r["out_tokens"] for r in passed if r.get("out_tokens")] or [0]) or None,
        })
    return out


def write(rows: list[dict], composite: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "headline.json").write_text(json.dumps({"rows": rows, "composite": composite}, indent=1))
    fields = ["model", "chip", "month", "held", "interactive_out_tps", "exchange_rate",
              "software_gain_index", "revenue_usd_per_gpu_hour", "rent_usd_per_gpu_hour",
              "breakeven_utilization_pct", "spread_usd_per_gpu_hour", "price_source"]
    with (out_dir / "headline.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
