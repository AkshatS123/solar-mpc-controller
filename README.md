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

**Phase 2 — runs end-to-end.** Synthetic trace, real cvxpy MPC, both
baselines, rolling-horizon simulator, working CLI, 8 passing tests.

```text
$ uv run python scripts/run_demo.py
controller   cost ($)   final SoC   deadline
MPC             3.649       0.799      False
Greedy          6.991       1.000       True
TOU            19.054       1.000       True
```

(Synthetic 1-day TOU trace, 60 kWh battery, 32 A max, 15-min step.
"Deadline False" for MPC = it satisfies SoC ≥ 0.8 at the deadline step but
not at trace end — see `docs/algorithm.md`.)

## Roadmap

- [x] **Phase 1** — synthetic trace + cvxpy MPC formulation
- [x] **Phase 2** — rolling-horizon simulator, baselines, CLI
- [ ] **Phase 3** — plots in `notebooks/`, real Tesphase trace replay (anonymized)
- [ ] **Phase 4** — quantile-forecast inputs, robust formulation
- [ ] **Phase 5** — writeup on [akshatsharma.blog](https://akshatsharma.blog)

## License

MIT — see [`LICENSE`](./LICENSE).
