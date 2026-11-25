# Compute conversion table

Measures how much finished AI work one GPU-hour produces on each chip, what that work sells for,
and how both compare with the GPU's rent. The full method is in the spec:
[Conversion Table Spec v0.1](https://claude.ai/artifact/SJmH1eCbduMEoW1Mf8iFCe).

Three headline numbers per chip, every month:

| Headline | What it says |
| --- | --- |
| Exchange rate | How many H100-hours one hour of the chip replaces |
| Break-even utilization | Share of hours a GPU must be busy at today's token prices to pay its rent |
| Software gain | How much more work the same H100 does than in the base month |

## How it works

| Part | Source | Runs |
| --- | --- | --- |
| Output per GPU | [SemiAnalysis InferenceX](https://inferencex.semianalysis.com/) public benchmark feed (vLLM runs) | Monthly |
| Rent | [GetDeploying GPU price dataset](https://getdeploying.com/dataset/gpu-prices), on-demand monthly median | Monthly |
| List prices | OpenRouter's public endpoints API, per provider | Daily |
| Provider field probes | Our own tasks, sent through OpenRouter and pinned to one provider at a time | 4 times a day |

The probes send the same synthetic tasks (with exact answer keys) to every provider of a model and record
cost, pass or fail, time to first token, output speed, errors and rate limits. Tasks come from a private
seed, so providers cannot recognise them; prompts are never stored, only task ids.

## Setup (once)

1. Create an [OpenRouter](https://openrouter.ai) account, add about $10 of credit, and create an API key
   with a monthly limit of about $40.
2. In this repository on GitHub: **Settings > Secrets and variables > Actions > New repository secret**:
   - `OPENROUTER_API_KEY`: the key from step 1.
   - `GRAIN_PRIVATE_SEED`: any long random phrase. Keep it secret; it generates the private tasks.
3. **Actions** tab: enable workflows if asked, then run **probes** once by hand (**Run workflow**) to check it.

Never paste the key into code, issues or chat.

## Commands

```bash
python -m grain prices            # today's list-price snapshot
python -m grain probe             # one probe slot (needs OPENROUTER_API_KEY)
python -m grain monthly           # InferenceX + rent + headline numbers
python -m grain summarize --month 2026-12   # provider field summary for a month
python -m grain show-task --type T2         # look at a public task
python -m unittest discover -s tests        # tests (no network or key needed)
```

Standard library only; Python 3.11 or newer.

## Data layout

| Path | Contents |
| --- | --- |
| `data/probes/YYYY/MM/YYYY-MM-DD_HH.jsonl` | One row per probe task |
| `data/prices/YYYY/MM/YYYY-MM-DD.json` | List prices per provider endpoint |
| `data/inferencex/` | Monthly output per GPU by chip, per model |
| `data/rent/getdeploying_monthly.json` | Monthly rent by chip |
| `data/headline/headline.csv` | The headline numbers |
| `data/field/YYYY-MM.json` | Provider field summary (price per Grain, pass rate, failures) |

## Costs

About $25-35 a month of OpenRouter credit (600 tasks a day) and about 700 GitHub Actions minutes a
month, inside the free allowance for private repositories.

## Caveats

- InferenceX runs are vendor- and community-tuned single-node benchmarks with synthetic 1,024-in /
  1,024-out requests; one request stands in for one Grain until probe token counts replace it.
- After the last InferenceX month for a model, output is held flat for up to 6 months and marked `held`.
- Probe timings include OpenRouter's routing overhead and come from GitHub's US runners.

## Licences and attribution

- Rent data: GetDeploying, CC BY 4.0.
- InferenceX: code is Apache 2.0; terms for its data feed are not stated, so ask SemiAnalysis before
  any commercial use.
- This project's published data: free for research use with attribution; commercial use needs a licence.
