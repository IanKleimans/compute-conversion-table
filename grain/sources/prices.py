"""Daily snapshot of OpenRouter's public list prices for each pinned model, per provider endpoint."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from . import fetch_json

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"


def snapshot(models_cfg: dict, out_dir: Path, day: dt.date, skip_if_exists: bool = False,
             fetch=fetch_json) -> Path | None:
    out = out_dir / "prices" / f"{day:%Y}" / f"{day:%m}" / f"{day:%Y-%m-%d}.json"
    if skip_if_exists and out.exists():
        return None
    result = {"date": day.isoformat(), "source": "openrouter.ai endpoints API", "models": {}}
    for model in models_cfg["models"]:
        data = fetch(ENDPOINTS_URL.format(slug=model["openrouter"])).get("data", {})
        endpoints = []
        for e in data.get("endpoints", []):
            pricing = e.get("pricing") or {}
            endpoints.append({
                "provider": e.get("provider_name"),
                "tag": e.get("tag"),
                "in_per_m": round(float(pricing.get("prompt") or 0) * 1e6, 4),
                "out_per_m": round(float(pricing.get("completion") or 0) * 1e6, 4),
                "quantization": e.get("quantization"),
                "status": e.get("status"),
                "uptime_last_30m": e.get("uptime_last_30m"),
                "context_length": e.get("context_length"),
            })
        endpoints.sort(key=lambda r: (r["out_per_m"], r["tag"] or ""))
        result["models"][model["code"]] = {"slug": model["openrouter"], "endpoints": endpoints}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True))
    return out


def median_list_price(snapshot_data: dict, model_code: str, providers: list[str]) -> tuple[float, float] | None:
    """Median input and output price per million tokens across the probed providers."""
    rows = [e for e in snapshot_data["models"].get(model_code, {}).get("endpoints", [])
            if (e.get("tag") or "").split("/")[0] in providers]
    if not rows:
        return None

    def median(values: list[float]) -> float:
        values = sorted(values)
        mid = len(values) // 2
        return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2

    return median([r["in_per_m"] for r in rows]), median([r["out_per_m"] for r in rows])


def latest_snapshot(out_dir: Path) -> dict | None:
    files = sorted((out_dir / "prices").glob("*/*/*.json"))
    return json.loads(files[-1].read_text()) if files else None
