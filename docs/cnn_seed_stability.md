# Satellite nowcaster (CNN): seed stability

The committed model (`models/nowcast_model.pt`, seed 0) was trained once, so its
metrics in `models/nowcast_model_metrics.json` come from a single seed. To check
how much they depend on that seed, `weather.nowcast_model.train(seed=s)` was run
for seeds 0, 1 and 2. Splits, data and hyperparameters were identical; outputs were
written outside the repo, and the committed model and metrics were not changed.

Each run took 2–3 minutes on a 4-core CPU with AVX2, with peak RSS of 2.4 GB.

## Skill vs persistence, test 2025 (squared-error skill, higher is better)

| Region | Seed 0 (committed) | Seed 1 | Seed 2 | Seed 0 rerun* | Range |
|---|---|---|---|---|---|
| north | 0.079 | 0.175 | 0.181 | 0.091 | 0.079 – 0.181 |
| south | 0.191 | 0.166 | 0.149 | 0.186 | 0.149 – 0.191 |
| east | 0.077 | 0.161 | 0.016 | 0.106 | 0.016 – 0.161 |
| **west** | **−0.385** | **−0.165** | **−0.280** | **−0.433** | always negative |
| central | 0.024 | 0.173 | 0.096 | 0.036 | 0.024 – 0.173 |
| northeast | 0.140 | 0.114 | 0.152 | 0.142 | 0.114 – 0.152 |

\* Retrain of seed 0 in the project venv on 2026-11-03 (step 9 of the final pass).
It was **not** bit-identical to the committed metrics, so it was reverted; see
"Reproducibility" below.

## Skill vs persistence, validation 2024

| Region | Seed 0 (committed) | Seed 1 | Seed 2 |
|---|---|---|---|
| north | 0.306 | 0.321 | 0.306 |
| south | 0.203 | 0.236 | 0.200 |
| east | 0.228 | 0.255 | 0.129 |
| west | −0.197 | −0.243 | −0.276 |
| central | 0.060 | 0.075 | −0.163 |
| northeast | −0.077 | 0.058 | −0.050 |

## Skill vs climatology, test 2025

| Region | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| north | 0.423 | 0.483 | 0.487 |
| south | 0.511 | 0.496 | 0.486 |
| east | 0.272 | 0.338 | 0.224 |
| west | 0.375 | 0.474 | 0.422 |
| central | 0.261 | 0.373 | 0.315 |
| northeast | 0.527 | 0.513 | 0.534 |

## What this shows

- **West India never beats persistence** (−0.165 to −0.433 on test). This is a
  consistent weakness, not bad luck with one seed.
- **Northeast correction.** The committed northeast test skill vs persistence is
  +0.140. The −0.077 sometimes quoted is the 2024 validation split, and validation
  northeast is negative on 2 of 3 seeds.
- **Spread between seeds is large.** It reaches 0.15 for east and central. Treat
  single-seed regional differences smaller than that as noise.
- **Climatology is beaten in every region on every seed** (0.22–0.53).

## Training curve of the committed model (seed 0)

The seed-0 run above reproduced the committed metrics exactly, so this is the
committed model's curve. Losses are MSE on log1p(mm). Early stopping restored the
best validation epoch (20) and stopped after 8 epochs without improvement.

| Epoch | Train | Val | | Epoch | Train | Val |
|---|---|---|---|---|---|---|
| 1 | 0.8833 | 0.5952 | | 15 | 0.1684 | 0.2468 |
| 2 | 0.4292 | 0.3721 | | 16 | 0.1646 | 0.2424 |
| 3 | 0.3194 | 0.3144 | | 17 | 0.1626 | 0.2416 |
| 4 | 0.2876 | 0.3043 | | 18 | 0.1580 | 0.2399 |
| 5 | 0.2529 | 0.2936 | | 19 | 0.1530 | 0.2418 |
| 6 | 0.2388 | 0.2768 | | **20** | **0.1463** | **0.2312** |
| 7 | 0.2273 | 0.3649 | | 21 | 0.1413 | 0.2402 |
| 8 | 0.2206 | 0.2591 | | 22 | 0.1395 | 0.2681 |
| 9 | 0.2197 | 0.2673 | | 23 | 0.1330 | 0.2363 |
| 10 | 0.2153 | 0.2589 | | 24 | 0.1290 | 0.2435 |
| 11 | 0.2073 | 0.2454 | | 25 | 0.1292 | 0.2548 |
| 12 | 0.1912 | 0.2514 | | 26 | 0.1228 | 0.2376 |
| 13 | 0.1881 | 0.2614 | | 27 | 0.1157 | 0.2470 |
| 14 | 0.1752 | 0.2410 | | 28 | 0.1149 | 0.2476 |

From epoch 20 on, train loss keeps falling (0.146 → 0.115) while validation loss
stops improving (best 0.231). This is moderate overfitting, which early stopping
contains. The model has no dropout, batch norm or augmentation.

Future training runs record this curve in `metrics["history"]`
(`weather/nowcast_model.py`). The committed metrics JSON predates that change and
has no `history` field.

## Reproducibility

Training is deterministic **when the dataset is read from the cache**. Three
seed-0 runs (two venvs, `uv.lock`-identical packages, 4 threads) reproduced the
committed weights and every metric bit-for-bit.

The one run that differed is the step-9 retrain, which also **rebuilt** the cache:
its test-2025 skill moved by up to 0.05 per region (column marked * above). The
rebuilt cache's X, Y and dates are byte-identical to the old cache, so the input
data is not the cause. The exact cause (likely floating-point nondeterminism on
that code path) was **not isolated**. That retrain was reverted, and the committed
`models/nowcast_model.pt` and `models/nowcast_model_metrics.json` are unchanged.

Practical consequence: a fresh rebuild of the CNN can move per-region skill by
about ±0.05 even with the same seed. That is smaller than the seed-to-seed spread
above.
