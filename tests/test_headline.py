import unittest

from grain import headline as hl
from grain.sources.inferencex import monthly_output
from grain.sources.rent import monthly_rent


def bench(chip, date, out_tps, user_tps, ttft=0.5, engine="vllm"):
    return {"benchmark_type": "single_turn", "is_multinode": False, "hardware": chip, "framework": engine,
            "precision": "fp4", "date": date, "image": "vllm/vllm-openai:v0.15.1",
            "metrics": {"output_tput_per_gpu": out_tps, "tput_per_gpu": 2 * out_tps,
                        "median_intvty": user_tps, "p99_ttft": ttft}}


MODELS = {"models": [{"code": "M", "providers": ["together"],
                      "list_price_fallback": {"in_per_m": 0.15, "out_per_m": 0.60}}]}


class InferenceXTest(unittest.TestCase):
    def test_speed_target_and_monthly_median(self):
        rows = [bench("h100", "2026-05-01", 3000, 40),   # too slow per user: batch only
                bench("h100", "2026-05-01", 1500, 120),
                bench("h100", "2026-05-02", 1600, 110),
                bench("h100", "2026-05-03", 1700, 105, ttft=5),  # first token too slow
                bench("h100", "2026-05-03", 1400, 150),
                bench("h100", "2026-05-03", 9000, 500, engine="trt")]  # other engine ignored
        [m] = monthly_output(rows)
        self.assertEqual(m["interactive_out_tps"], 1500)  # median of 1500, 1600, 1400
        self.assertEqual(m["batch_out_tps"], 1700)        # median of 3000, 1600, 1700
        self.assertEqual(m["days"], 3)


class HeadlineTest(unittest.TestCase):
    def test_reproduces_spec_number(self):
        # gpt-oss-120b on H100: May 2026 output held to November, rent $4.07 -> 98% break-even (spec chart).
        ix = {"M": [{"chip": "h100", "month": "2026-05", "interactive_out_tps": 1572.2, "batch_out_tps": 2071.1}]}
        rent = [{"chip": "h100", "month": "2026-11", "usd_per_gpu_hour": 4.07}]
        rows = hl.compute(MODELS, ix, rent, snapshots={})
        nov = next(r for r in rows if r["month"] == "2026-11")
        self.assertTrue(nov["held"])
        self.assertEqual(nov["breakeven_utilization_pct"], 98)
        self.assertEqual(nov["revenue_usd_per_gpu_hour"], 4.16)

    def test_hold_is_capped(self):
        ix = {"M": [{"chip": "h100", "month": "2025-10", "interactive_out_tps": 1000, "batch_out_tps": 1000}]}
        rent = [{"chip": "h100", "month": "2026-11", "usd_per_gpu_hour": 4.0}]
        months = [r["month"] for r in hl.compute(MODELS, ix, rent, {})]
        self.assertEqual(months[-1], "2026-04")

    def test_exchange_rate_and_composite(self):
        ix = {"M": [{"chip": "h100", "month": "2026-02", "interactive_out_tps": 1000, "batch_out_tps": 1},
                    {"chip": "b200", "month": "2026-02", "interactive_out_tps": 4100, "batch_out_tps": 1}]}
        rows = hl.compute(MODELS, ix, [], {})
        b200 = next(r for r in rows if r["chip"] == "b200")
        self.assertEqual(b200["exchange_rate"], 4.1)
        self.assertEqual(hl.composite_exchange_rates(rows)[0]["composite_exchange_rate"], 4.1)


class RentTest(unittest.TestCase):
    def test_monthly_median(self):
        csv = ("gpu_slug,date,billing_type,reservation_months,min_price,max_price,median_price,"
               "provider_median_price,provider_count,offering_count\n"
               "nvidia-h100,2026-11-02,ON_DEMAND,,1,9,4.0,4,40,100\n"
               "nvidia-h100,2026-11-09,ON_DEMAND,,1,9,4.2,4,40,100\n"
               "nvidia-h100,2026-11-16,ON_DEMAND,,1,9,4.1,4,40,100\n"
               "nvidia-h100,2026-11-16,RESERVATION,,1,9,2.0,4,40,100\n")
        self.assertEqual(monthly_rent(csv), [{"chip": "h100", "month": "2026-11", "usd_per_gpu_hour": 4.1, "weeks": 3}])


if __name__ == "__main__":
    unittest.main()
