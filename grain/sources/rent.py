"""Monthly GPU rent from GetDeploying's weekly dataset (CC BY 4.0, attribution required).

Source: https://getdeploying.com/dataset/gpu-prices
Uses the on-demand weekly median across providers, reduced to a monthly median.
"""

from __future__ import annotations

import collections
import csv
import io
import statistics

from . import fetch

WEEKLY_CSV = "https://getdeploying.com/dataset/gpu-prices/weekly.csv"
SLUGS = {"nvidia-a100": "a100", "nvidia-h100": "h100", "nvidia-h200": "h200", "nvidia-b200": "b200"}


def monthly_rent(csv_text: str | None = None) -> list[dict]:
    text = csv_text if csv_text is not None else fetch(WEEKLY_CSV).decode()
    weeks = collections.defaultdict(list)
    for row in csv.DictReader(io.StringIO(text)):
        chip = SLUGS.get(row["gpu_slug"])
        if chip and row["billing_type"] == "ON_DEMAND" and row["median_price"]:
            weeks[(chip, row["date"][:7])].append(float(row["median_price"]))
    return [{"chip": chip, "month": month, "usd_per_gpu_hour": round(statistics.median(v), 2), "weeks": len(v)}
            for (chip, month), v in sorted(weeks.items())]
