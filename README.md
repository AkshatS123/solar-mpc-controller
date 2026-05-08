# solar-mpc-controller

> Open-source model-predictive controller for charging an EV from rooftop solar.
>
> Receding-horizon optimization in [`cvxpy`](https://www.cvxpy.org/), with greedy
> and rule-based baselines for honest comparison.

## The question

Given a solar production forecast, a load forecast, a tariff schedule, and a
battery state of charge, **what charging amperage trajectory minimizes grid
import cost while respecting the EV's session deadline?**

Most home energy products answer this with greedy rules ("charge when export
exceeds 1 kW"). MPC reframes it as a constrained optimization over a rolling
horizon and re-solves every step as new measurements arrive. This repo is the
public reference implementation behind the v2 charging algorithm at
[Tesphase](https://tesphase.co).

## Approach

At each control step `t` we solve:

```
minimize    Σ  c_grid(τ) · max(0, load(τ) + charge(τ) − solar(τ))
            τ=t..t+H
            + λ_smooth · ||Δ charge||²
            − λ_soc · SoC(t+H)

subject to  0 ≤ charge(τ) ≤ amp_max
            charge(τ) ∈ feasible_amp_set        (EV step granularity)
            SoC(t+H) ≥ SoC_target_by_deadline
            |charge(τ+1) − charge(τ)| ≤ Δ_max
```

Inputs come from any forecaster — point or quantile. The companion benchmark
[`solar-tsfm-bench`](https://github.com/AkshatS123/solar-tsfm-bench) provides
forecast traces this controller can consume.

## Why this exists

1. **An open MPC reference for residential EV charging.** Most published MPC
   work targets utility-scale storage; the residential case has different
   constraints (discrete amperage, OEM API latencies, partial observability).
2. **A teacher policy for imitation learning.** Distill the MPC into a small
   policy network for fast on-device inference — handled in
   [`tsfm-imitation-policy`](https://github.com/AkshatS123/tsfm-imitation-policy)
   _(planned)_.
3. **Honest baselines.** Greedy and rule-based controllers are first-class
   citizens here — the goal is to measure the gap, not to hide it.

## Repo layout

```
solar-mpc-controller/
├── src/solar_mpc/
│   ├── controller.py     # cvxpy MPC formulation
│   ├── baselines.py      # greedy + rule-based controllers
│   ├── simulator.py      # rolling-horizon driver
│   ├── traces.py         # synthetic + recorded trace loader
│   └── cli.py            # `solar-mpc run …`
├── tests/                # pytest, runs in CI
├── scripts/run_demo.py   # one-command end-to-end demo
├── notebooks/            # walkthroughs and result plots
├── data/synthetic/       # tiny synthetic traces checked in
└── docs/algorithm.md     # math + assumptions writeup
```

## Quickstart

```bash
brew install uv          # if needed
uv sync
uv run pytest -q
uv run python scripts/run_demo.py
uv run solar-mpc --help
```

## Status

**Phase 4 — robust under noise.** Synthetic trace, real cvxpy MPC,
scenario-based RobustMPC with optional CVaR, mixed-integer amperage
variant, rolling-horizon simulator with stochastic realizations, both
heuristic baselines, working CLI, 28 passing tests, three figures and a
two-part walkthrough.

### Perfect-forecast comparison

| Controller | Strategy                              | Cost ($) | Final SoC |
|------------|---------------------------------------|----------|-----------|
| **MPC**    | Receding-horizon optimization (cvxpy) | **6.81** |   0.917   |
| Greedy     | Charge on solar excess                | 7.67     |   1.000   |
| TOU        | Charge on cheap price                 | 10.82    |   1.000   |

![comparison](./notebooks/figures/02_comparison.png)

### Reliability under noisy forecasts (50 realizations, σ_solar=0.40)

| Controller | Cost ($) | **Deadline miss** |
|------------|----------|-------------------|
| **MPC**         | 9.91 ± 0.18 | **0 %** |
| RobustMPC       | 9.94 ± 0.18 | **0 %** |
| Greedy          | 8.65 ± 0.32 | **100 %** |
| TOU             | 12.52 ± 0.12 | **0 %** |

![reliability](./notebooks/figures/03_robustness.png)

Headline: solar-greedy is the cheapest controller — and fails at the
deadline 100 % of the time. MPC reliably succeeds. See the [walkthrough](./notebooks/01_walkthrough.md)
for why nominal MPC is already robust on this problem and when robust
extensions would dominate.

→ Read the walkthrough: [`notebooks/01_walkthrough.md`](./notebooks/01_walkthrough.md)
→ Read the algorithm spec: [`docs/algorithm.md`](./docs/algorithm.md)

## Roadmap

- [x] **Phase 1** — synthetic trace + cvxpy MPC formulation
- [x] **Phase 2** — rolling-horizon simulator, baselines, CLI
- [x] **Phase 3** — figures, notebook walkthrough, algorithm writeup
- [x] **Phase 4** — stochastic simulator, RobustMPC, CVaR variant, MIP variant, tsfm-bench adapter shim
- [x] **Phase 5 (engineering)** — `scripts/replay_tesphase.py`, `docs/blog_draft.md`, `docs/launch_assets.md`
- [ ] **Phase 5 (publishing)** — Tesphase own-house trace export, blog post on [akshatsharma.blog](https://akshatsharma.blog), Twitter / LinkedIn / cold emails, `v0.1.0` release tag

## License

MIT — see [`LICENSE`](./LICENSE).
