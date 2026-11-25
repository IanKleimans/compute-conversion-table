"""Deterministic synthetic task suite.

Every task is generated from (seed, index), so the same pair always gives the same
prompt and answer key. Probes use a private seed that providers cannot know; the public
seed lets anyone regenerate the public suite.

Three task types, matching the spec:
  T1  short answer over a ~1,000-token product catalog
  T2  field extraction from a ~16,000-token shipment log
  T3  multi-step stock arithmetic over a ~1,500-token ledger

Lengths are targeted with a rough 4 characters per token; real token counts come back
from the provider's usage report and are recorded with each probe.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass

PUBLIC_SEED = "public-v0.1"
CHARS_PER_TOKEN = 4.0

ADJECTIVES = [
    "Amber", "Brisk", "Cobalt", "Dusky", "Ember", "Frost", "Granite", "Harbor", "Ivory",
    "Juniper", "Kestrel", "Lumen", "Marble", "Nimbus", "Onyx", "Pine", "Quartz", "Russet",
    "Sable", "Tidal", "Umber", "Velvet", "Willow", "Zephyr",
]
NOUNS = [
    "Kettle", "Lantern", "Satchel", "Compass", "Blender", "Easel", "Thermos", "Router",
    "Tripod", "Monitor", "Stapler", "Backpack", "Headset", "Speaker", "Drill", "Ladder",
]
COLORS = ["red", "blue", "green", "black", "white", "grey", "yellow", "orange"]
WAREHOUSES = ["North", "South", "East", "West", "Central"]
CITIES = [
    "Aldport", "Brenvale", "Corrin", "Dunmore", "Eastwick", "Farrow", "Glenhaven", "Holloway",
    "Islington", "Jarrow", "Kingsreach", "Lowmoor", "Marlow", "Northam", "Oakridge", "Pembry",
]
STATUSES = ["in transit", "delivered", "held at customs", "awaiting pickup", "returned"]
ITEMS = ["bolts", "gaskets", "valves", "sensors", "cables", "filters"]


@dataclass(frozen=True)
class Task:
    id: str
    type: str
    prompt: str
    answer: str
    kind: str  # "number", "text" or "json"
    target_input_tokens: int


def _rng(seed: str, index: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{index}".encode()).hexdigest()
    return random.Random(int(digest, 16))


def _fill(lines_fn, target_chars: int) -> list[str]:
    lines: list[str] = []
    size = 0
    while size < target_chars:
        line = lines_fn(len(lines))
        lines.append(line)
        size += len(line) + 1
    return lines


def make_t1(rng: random.Random, task_id: str, target_tokens: int) -> Task:
    used: set[str] = set()
    products = []

    def line(i: int) -> str:
        while True:
            name = f"{rng.choice(ADJECTIVES)} {rng.choice(NOUNS)} {rng.randint(100, 999)}"
            if name not in used:
                used.add(name)
                break
        p = {
            "name": name,
            "sku": f"SKU-{rng.randint(10000, 99999)}",
            "color": rng.choice(COLORS),
            "weight": round(rng.uniform(0.2, 40.0), 1),
            "price": round(rng.uniform(5, 900), 2),
            "warehouse": rng.choice(WAREHOUSES),
        }
        products.append(p)
        return (f"- {p['name']} | {p['sku']} | colour {p['color']} | {p['weight']} kg | "
                f"${p['price']:.2f} | warehouse {p['warehouse']}")

    header = "Product catalog (name | SKU | colour | weight | price | warehouse):\n"
    lines = _fill(line, int(target_tokens * CHARS_PER_TOKEN * 0.9) - len(header))
    catalog = header + "\n".join(lines)

    template = rng.choice(["lookup", "count", "max"])
    if template == "lookup":
        p = rng.choice(products)
        question = (f'What is the weight in kilograms of the product named "{p["name"]}"? '
                    "Reply with the number only.")
        answer, kind = f"{p['weight']}", "number"
    elif template == "count":
        w = rng.choice(WAREHOUSES)
        n = sum(1 for p in products if p["warehouse"] == w)
        question = (f"How many products in the catalog are stored in warehouse {w}? "
                    "Reply with the number only.")
        answer, kind = str(n), "number"
    else:
        while True:
            w = rng.choice(WAREHOUSES)
            stock = sorted((p for p in products if p["warehouse"] == w), key=lambda p: -p["price"])
            if len(stock) >= 2 and stock[0]["price"] != stock[1]["price"]:
                break
        question = (f"Which product stored in warehouse {w} has the highest price? "
                    "Reply with the product name only.")
        answer, kind = stock[0]["name"], "text"

    prompt = f"{catalog}\n\n{question}"
    return Task(task_id, "T1", prompt, answer, kind, target_tokens)


def make_t2(rng: random.Random, task_id: str, target_tokens: int) -> Task:
    records = []
    used: set[str] = set()

    def line(i: int) -> str:
        while True:
            trk = f"TRK-{rng.randint(100000, 999999)}"
            if trk not in used:
                used.add(trk)
                break
        r = {
            "time": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d} "
                    f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}",
            "shipment_id": f"SHP-{rng.randint(1000000, 9999999)}",
            "tracking": trk,
            "origin": rng.choice(CITIES),
            "dest": rng.choice(CITIES),
            "status": rng.choice(STATUSES),
            "weight_kg": round(rng.uniform(1, 2500), 1),
        }
        records.append(r)
        return (f"{r['time']} | {r['shipment_id']} | tracking={r['tracking']} | origin={r['origin']} | "
                f"dest={r['dest']} | status={r['status']} | weight_kg={r['weight_kg']}")

    header = "Shipment log, one shipment per line:\n"
    lines = _fill(line, int(target_tokens * CHARS_PER_TOKEN * 0.95) - len(header))
    # Pick a record away from both ends so position is not a giveaway.
    target = records[rng.randint(len(records) // 10, len(records) - len(records) // 10 - 1)]
    question = (f"Find the shipment with tracking code {target['tracking']}. Return only a JSON object "
                'with the keys "shipment_id", "dest" and "weight_kg" (weight as a number).')
    answer = json.dumps({"shipment_id": target["shipment_id"], "dest": target["dest"],
                         "weight_kg": target["weight_kg"]}, sort_keys=True)
    prompt = header + "\n".join(lines) + "\n\n" + question
    return Task(task_id, "T2", prompt, answer, "json", target_tokens)


def make_t3(rng: random.Random, task_id: str, target_tokens: int) -> Task:
    opening = {item: rng.randint(50, 400) for item in ITEMS}
    stock = dict(opening)

    def line(i: int) -> str:
        item = rng.choice(ITEMS)
        if rng.random() < 0.5 or stock[item] < 20:
            qty = rng.randint(5, 120)
            stock[item] += qty
            verb = "received"
        else:
            qty = rng.randint(1, min(stock[item], 120))
            stock[item] -= qty
            verb = "shipped"
        return (f"Entry {i + 1}: day {i // 3 + 1}, {verb} {qty} units of {item} "
                f"(batch B{rng.randint(100, 999)}, handled by team {rng.choice(WAREHOUSES)}).")

    head = ("Opening stock: " + ", ".join(f"{v} {k}" for k, v in opening.items()) + ".\n"
            "Stock ledger:\n")
    lines = _fill(line, int(target_tokens * CHARS_PER_TOKEN * 0.9) - len(head))
    item = rng.choice(ITEMS)
    question = (f"Using the opening stock and every ledger entry, what is the closing quantity of {item}? "
                "Work it out, then end your reply with a final line of the form ANSWER: <number>.")
    prompt = head + "\n".join(lines) + "\n\n" + question
    return Task(task_id, "T3", prompt, str(stock[item]), "number", target_tokens)


MAKERS = {"T1": make_t1, "T2": make_t2, "T3": make_t3}


def make_task(seed: str, index: int, type_cycle: list[str], targets: dict[str, int],
              label: str = "task") -> Task:
    """Task number `index` of the suite: its type follows `type_cycle`, its content the seed."""
    task_type = type_cycle[index % len(type_cycle)]
    rng = _rng(seed, index)
    return MAKERS[task_type](rng, f"{label}-{index}", targets[task_type])
