"""Command line: python -m grain <command>.

  prices      snapshot today's OpenRouter list prices for the pinned models
  probe       run one probe slot (needs OPENROUTER_API_KEY; GRAIN_PRIVATE_SEED recommended)
  inferencex  pull InferenceX history and write monthly output per GPU
  rent        pull GetDeploying weekly rent and write monthly medians
  headline    compute the headline numbers from the files above
  monthly     inferencex + rent + headline
  summarize   provider field summary (Table D) for one month of probes
  show-task   print one public task, to see what providers are asked
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from . import CONFIG, DATA, ROOT
from . import headline as hl
from .probe import load_json, nearest_slot, run_slot
from .sources import inferencex, prices, rent
from .tasks import PUBLIC_SEED, make_task


def _today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def cmd_prices(args) -> int:
    path = prices.snapshot(load_json(CONFIG / "models.json"), DATA, _today(), skip_if_exists=args.once_per_day)
    print(path or "today's snapshot already exists")
    return 0


def cmd_probe(args) -> int:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY is not set; probes skipped", file=sys.stderr)
        return 0 if args.skip_without_key else 2
    probe_cfg = load_json(CONFIG / "probe.json")
    now = dt.datetime.now(dt.timezone.utc)
    day = dt.date.fromisoformat(args.date) if args.date else now.date()
    slot = args.slot if args.slot is not None else nearest_slot(now.hour, probe_cfg["slots_utc"])
    run_slot(load_json(CONFIG / "models.json"), probe_cfg, key,
             os.environ.get("GRAIN_PRIVATE_SEED") or None, day, slot, DATA)
    return 0


def cmd_inferencex(args) -> int:
    targets = load_json(CONFIG / "probe.json")["interactive"]["min_tokens_per_s"]
    for model in load_json(CONFIG / "models.json")["models"]:
        if not model.get("inferencex"):
            continue
        min_tps = args.min_user_tps or targets[model["code"]]
        history = inferencex.fetch_history(model["inferencex"])
        monthly = inferencex.monthly_output(history, precision=model.get("inferencex_precision"),
                                            min_user_tps=min_tps)
        out = DATA / "inferencex" / f"{model['code']}_{model['inferencex']}_1024x1024.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"source": inferencex.HISTORY_URL, "model": model["inferencex"],
                                   "min_user_tps": min_tps, "engine": "vllm",
                                   "rows": monthly}, indent=1))
        print(f"{model['code']}: {len(monthly)} chip-months -> {out}")
    return 0


def cmd_rent(args) -> int:
    rows = rent.monthly_rent()
    out = DATA / "rent" / "getdeploying_monthly.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": rent.WEEKLY_CSV, "licence": "CC BY 4.0, GetDeploying",
                               "rows": rows}, indent=1))
    print(f"{len(rows)} chip-months -> {out}")
    return 0


def cmd_headline(args) -> int:
    models_cfg = load_json(CONFIG / "models.json")
    ix = {}
    for f in (DATA / "inferencex").glob("*.json"):
        ix[f.name.split("_")[0]] = json.loads(f.read_text())["rows"]
    rent_rows = json.loads((DATA / "rent" / "getdeploying_monthly.json").read_text())["rows"]
    snapshots = {json.loads(f.read_text())["date"]: json.loads(f.read_text())
                 for f in (DATA / "prices").glob("*/*/*.json")}
    rows = hl.compute(models_cfg, ix, rent_rows, snapshots)
    composite = hl.composite_exchange_rates(rows)
    hl.write(rows, composite, DATA / "headline")
    latest = max((r["month"] for r in rows), default=None)
    print(f"{len(rows)} rows written to {DATA / 'headline'}; latest month {latest}")
    for r in rows:
        if r["month"] == latest:
            print(f"  {r['model']} {r['chip']:<5} break-even {r.get('breakeven_utilization_pct', '-')}%"
                  f"{' (output held)' if r['held'] else ''}")
    return 0


def cmd_site(args) -> int:
    from .site import build
    site = build(load_json(CONFIG / "models.json"), DATA, ROOT / "docs" / "data" / "site.json", _today())
    print(f"site data written: {len(site['breakeven'])} break-even points, prices from {site['prices_date']}")
    return 0


def cmd_monthly(args) -> int:
    return cmd_inferencex(args) or cmd_rent(args) or cmd_headline(args) or cmd_site(args)


def cmd_summarize(args) -> int:
    rows = []
    for f in sorted((DATA / "probes" / args.month[:4] / args.month[5:7]).glob("*.jsonl")):
        rows += [json.loads(line) for line in f.read_text().splitlines() if line]
    summary = hl.field_summary(rows)
    out = DATA / "field" / f"{args.month}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    for s in summary:
        print(s)
    return 0


def cmd_show_task(args) -> int:
    cfg = load_json(CONFIG / "probe.json")
    cycle = cfg["type_cycle"]
    index = next(i for i in range(args.index, args.index + len(cycle)) if cycle[i % len(cycle)] == args.type)
    task = make_task(PUBLIC_SEED, index, cycle, cfg["target_input_tokens"], "public")
    print(task.prompt[:1500] + ("\n[...]\n" + task.prompt[-600:] if len(task.prompt) > 2100 else ""))
    print(f"\n--- answer key: {task.answer}  ({len(task.prompt)} characters)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grain", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prices"); p.add_argument("--once-per-day", action="store_true"); p.set_defaults(fn=cmd_prices)
    p = sub.add_parser("probe"); p.add_argument("--slot", type=int); p.add_argument("--date")
    p.add_argument("--skip-without-key", action="store_true",
                   help="exit quietly when no API key is set (used by the scheduled workflow)")
    p.set_defaults(fn=cmd_probe)
    for name, fn in (("inferencex", cmd_inferencex), ("monthly", cmd_monthly)):
        p = sub.add_parser(name)
        p.add_argument("--min-user-tps", type=float, default=None,
                       help="override the per-model speed target in config/probe.json")
        p.set_defaults(fn=fn)
    sub.add_parser("rent").set_defaults(fn=cmd_rent)
    sub.add_parser("headline").set_defaults(fn=cmd_headline)
    sub.add_parser("site").set_defaults(fn=cmd_site)
    p = sub.add_parser("summarize"); p.add_argument("--month", required=True, help="YYYY-MM"); p.set_defaults(fn=cmd_summarize)
    p = sub.add_parser("show-task"); p.add_argument("--type", default="T1", choices=["T1", "T2", "T3"])
    p.add_argument("--index", type=int, default=0); p.set_defaults(fn=cmd_show_task)
    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
