"""Build docs/data/site.json, the only file the public page (docs/index.html) reads."""

from __future__ import annotations

import datetime as dt
import json
import statistics
from pathlib import Path

from .sources.prices import latest_snapshot

# InferenceX launched on 29 Sep 2025, so September holds only two days of runs.
START_MONTH = "2025-10"
CHART_MODEL = "M"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def build(models_cfg: dict, data_dir: Path, out_path: Path, today: dt.date) -> dict:
    headline = json.loads((data_dir / "headline" / "headline.json").read_text())
    rows = [r for r in headline["rows"] if r["month"] >= START_MONTH]
    models = {m["code"]: m for m in models_cfg["models"]}
    chart = [r for r in rows if r["model"] == CHART_MODEL]

    breakeven = [{"month": r["month"], "chip": r["chip"].upper(), "pct": r["breakeven_utilization_pct"],
                  "rent": r["rent_usd_per_gpu_hour"], "revenue": r["revenue_usd_per_gpu_hour"],
                  "held": r["held"]}
                 for r in chart if r.get("breakeven_utilization_pct") is not None]
    exchange = [{"month": r["month"], "chip": r["chip"].upper(), "rate": r["exchange_rate"]}
                for r in chart if r.get("exchange_rate") is not None and r["chip"] != "h100"]
    h100 = [r for r in chart if r["chip"] == "h100" and not r["held"]]
    base = h100[0] if h100 else None
    gain = [{"month": r["month"], "index": round(100 * r["interactive_out_tps"] / base["interactive_out_tps"], 1)}
            for r in h100] if base else []

    latest = []
    for chip in ("h100", "h200", "b200"):
        series = [r for r in chart if r["chip"] == chip and r.get("breakeven_utilization_pct") is not None]
        if series:
            r = series[-1]
            last_measured = max(x["month"] for x in chart if x["chip"] == chip and not x["held"])
            latest.append({"chip": chip.upper(), "month": r["month"], "held": r["held"],
                           "measured_month": last_measured,
                           "out_tps": r["interactive_out_tps"], "revenue": r["revenue_usd_per_gpu_hour"],
                           "rent": r["rent_usd_per_gpu_hour"], "pct": r["breakeven_utilization_pct"],
                           "spread": r["spread_usd_per_gpu_hour"]})

    prices = []
    snap = latest_snapshot(data_dir)
    if snap:
        for code in [m["code"] for m in models_cfg["models"] if m["code"] in snap["models"]]:
            m = snap["models"][code]
            outs = [e["out_per_m"] for e in m["endpoints"] if e["out_per_m"] > 0]
            prices.append({"model": models[code]["name"], "providers": len(outs),
                           "min": min(outs) if outs else None, "median": _median(outs),
                           "max": max(outs) if outs else None})

    site = {
        "updated": today.isoformat(),
        "prices_date": snap["date"] if snap else None,
        "model": {"code": CHART_MODEL, "name": models[CHART_MODEL]["name"],
                  "price_source": chart[-1]["price_source"] if chart else None},
        "breakeven": breakeven,
        "exchange": exchange,
        "gain": gain,
        "gain_base_month": base["month"] if base else None,
        "latest": latest,
        "prices": prices,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(site, indent=1))
    return site
