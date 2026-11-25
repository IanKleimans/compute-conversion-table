import json
import re
import unittest

from grain.grade import grade
from grain.tasks import CHARS_PER_TOKEN, PUBLIC_SEED, make_task

CYCLE = ["T1", "T2", "T1", "T3"]
TARGETS = {"T1": 1000, "T2": 16000, "T3": 1500}


def task(i, seed=PUBLIC_SEED):
    return make_task(seed, i, CYCLE, TARGETS)


class TaskSuiteTest(unittest.TestCase):
    def test_same_seed_same_task(self):
        self.assertEqual(task(7), task(7))

    def test_private_seed_changes_content(self):
        self.assertNotEqual(task(7).prompt, task(7, seed="secret").prompt)

    def test_type_follows_cycle(self):
        self.assertEqual([task(i).type for i in range(8)], CYCLE * 2)

    def test_lengths_near_target(self):
        for i in range(8):
            t = task(i)
            approx_tokens = len(t.prompt) / CHARS_PER_TOKEN
            self.assertGreater(approx_tokens, 0.85 * t.target_input_tokens, t.id)
            self.assertLess(approx_tokens, 1.15 * t.target_input_tokens, t.id)

    def test_t2_answer_is_in_the_log(self):
        for i in (1, 5, 9):
            t = task(i)
            want = json.loads(t.answer)
            line = next(l for l in t.prompt.splitlines() if want["shipment_id"] in l)
            self.assertIn(f"weight_kg={want['weight_kg']}", line)

    def test_t3_answer_matches_ledger(self):
        t = task(3)
        item = re.search(r"closing quantity of (\w+)", t.prompt).group(1)
        stock = int(re.search(rf"(\d+) {item}", t.prompt.split("\n")[0]).group(1))
        for verb, qty in re.findall(rf"(received|shipped) (\d+) units of {item}\b", t.prompt):
            stock += int(qty) if verb == "received" else -int(qty)
        self.assertEqual(str(stock), t.answer)


class GradingTest(unittest.TestCase):
    def test_number(self):
        t = task(3)
        self.assertTrue(grade(t, f"Adding it up...\nANSWER: {t.answer}"))
        self.assertFalse(grade(t, f"ANSWER: {int(t.answer) + 1}"))
        self.assertFalse(grade(t, ""))

    def test_json(self):
        t = task(1)
        want = json.loads(t.answer)
        self.assertTrue(grade(t, "Here you go: ```json\n" + json.dumps(want) + "\n```"))
        wrong = dict(want, weight_kg=want["weight_kg"] + 1)
        self.assertFalse(grade(t, json.dumps(wrong)))

    def test_text(self):
        t = next(task(i) for i in range(0, 400, 2) if task(i).kind == "text")
        self.assertTrue(grade(t, t.answer + "."))
        self.assertTrue(grade(t, f'"{t.answer}"'))
        self.assertFalse(grade(t, "Some other product 123"))


if __name__ == "__main__":
    unittest.main()
