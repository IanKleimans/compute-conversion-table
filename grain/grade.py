"""Pass/fail checks against a task's answer key."""

from __future__ import annotations

import json
import re

from .tasks import Task

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
JSON_OBJECT = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _numbers(text: str) -> list[float]:
    out = []
    for raw in NUMBER.findall(text):
        try:
            out.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def _norm(text: str) -> str:
    return re.sub(r"[\s\"'`*.]+", " ", text.lower()).strip()


def grade(task: Task, text: str | None) -> bool:
    if not text:
        return False
    if task.kind == "number":
        tail = text.rsplit("ANSWER:", 1)[1] if "ANSWER:" in text else text
        nums = _numbers(tail)
        if not nums:
            return False
        # After "ANSWER:" the first number is the answer; otherwise the last one in the reply.
        value = nums[0] if "ANSWER:" in text else nums[-1]
        return abs(value - float(task.answer)) <= 0.01
    if task.kind == "text":
        got, want = _norm(text), _norm(task.answer)
        return got == want or (want in got and len(got) <= len(want) + 40)
    if task.kind == "json":
        want = json.loads(task.answer)
        for candidate in JSON_OBJECT.findall(text):
            try:
                got = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            try:
                return (str(got.get("shipment_id")) == want["shipment_id"]
                        and _norm(str(got.get("dest"))) == _norm(want["dest"])
                        and abs(float(got.get("weight_kg")) - want["weight_kg"]) <= 0.01)
            except (TypeError, ValueError):
                return False
        return False
    raise ValueError(f"unknown answer kind {task.kind!r}")
