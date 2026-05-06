# Algorithm

Specification of the MPC formulation in `solar_mpc.controller.MPCController.step()`.

## Variables and indices

| Symbol | Meaning |
|--------|---------|
| τ ∈ {0, …, H−1} | Time index over the horizon |
| H | Horizon length (steps) |
| Δt | Step duration (hours) |
| a[τ] | Charging amperage at step τ — **decision variable** |
| p[τ] | Charging power (kW), derived: `p[τ] = a[τ] · V / 1000` |
| s[τ] | SoC at the end of step τ (fraction of capacity) |
| g[τ] | Grid import (kW) — derived |
| L[τ] | House load forecast (kW) — exogenous input |
| S[τ] | Solar production forecast (kW) — exogenous input |
| π[τ] | Grid price ($/kWh) — exogenous input |

## Parameters (`ControllerConfig`)

| Symbol | Field | Meaning |
|--------|-------|---------|
| V | `voltage` | Charger voltage (typ. 240 V) |
| C | `battery_capacity_kwh` | Battery capacity (kWh) |
| a_max | `amp_max` | Maximum amperage |
| Δa_max | `delta_amp_max` | Per-step amperage slew limit |
| s_target | `soc_target` | Required SoC at the deadline |
| τ_d | `inputs.deadline_step` | Deadline step index (within horizon, optional) |
| λ_smooth | `smoothness_weight` | Smoothness penalty weight |
| λ_soc | `soc_terminal_weight` | Terminal SoC reward weight |

## Objective

Minimize, over a[0..H−1]:

```
Σ_τ π[τ] · g[τ] · Δt          (energy cost)
  + λ_smooth · Σ_τ (a[τ+1] − a[τ])²    (smoothness penalty)
  − λ_soc · s[H−1]              (terminal SoC reward)
```

## Constraints

```
0 ≤ a[τ] ≤ a_max                     for all τ
|a[τ+1] − a[τ]| ≤ Δa_max              for τ < H−1
s[τ] ≤ 1.0                            for all τ
s[τ_d − 1] ≥ s_target                  if τ_d is within horizon
```

with derived equations

```
p[τ] = a[τ] · V / 1000
s[τ] = s_now + (Δt / C) · Σ_{k=0..τ} p[k]
g[τ] = max(0, L[τ] + p[τ] − S[τ])
```

## Why this is a convex problem

Every term above is convex in `a`:
- `g[τ] = pos(L + a·V/1000 − S)` is the pointwise max of two affine
  functions, which is convex.
- The energy cost is a non-negative weighted sum of convex terms.
- The smoothness term is `sum_squares(diff(a))`, quadratic and convex.
- The terminal SoC reward is linear (negated, since we minimize).
- All constraints are affine in `a`.

So the whole problem is a QP with conic constraints, solvable by CLARABEL.

## Solver

`cvxpy` ≥ 1.5 with CLARABEL (interior-point for convex conic).
SCS or ECOS would also work; CLARABEL is the current default for
its better behavior on QP + cone mixes.

## Infeasibility handling

If CLARABEL reports infeasible / unbounded (e.g., the deadline is
unreachable from the current SoC even at `a_max`), `step()` returns
`a_max` for the current step as a safe fallback rather than propagating
the infeasibility. The hope is that the *next* call — with a fresh
measurement and a shifted horizon — finds a feasible plan.

This is a deliberate design choice: a residential charging controller
should never refuse to act. A pathological-but-reasonable failure mode
(charge as fast as possible) is preferable to a stop.

## Receding horizon dynamics

At each control tick, the caller passes:
- the latest `s_now` (measured, not predicted),
- a fresh window of forecasts `L`, `S`, `π` aligned to this tick.

Only `a[0]` is applied. The rest of the plan is discarded and recomputed
next tick. This is the property that makes MPC robust to forecast error
in practice — wrong forecasts get corrected immediately when the
optimization re-solves.

## Trade-offs and known limitations

- **Continuous amperage relaxation.** Most EVs step in 1 A increments.
  Continuous `a[τ]` keeps the LP/QP convex; rounding at apply time
  introduces small SoC drift. Phase 4 will explore the mixed-integer
  formulation.
- **Point forecasts only.** No uncertainty quantification. Robust /
  scenario / chance-constrained formulations belong in Phase 4 once
  quantile inputs from `solar-tsfm-bench` are wired up.
- **No demand charges.** The cost model is pure energy ($/kWh).
  Adding a peak-demand term is straightforward (an `inf_norm` over a
  rolling window) but changes the solver story.
- **No battery degradation cost.** Charging hard looks free in this
  cost model. Real economic models include $/cycle.
- **Single vehicle.** No coordination across multiple EVs.

## References

- Borrelli, F., Bemporad, A., & Morari, M. (2017). *Predictive Control
  for Linear and Hybrid Systems*. Cambridge University Press.
- Diamond, S. & Boyd, S. (2016). CVXPY: A Python-embedded modeling
  language for convex optimization. *JMLR* 17 (83): 1–5.
- Goebel, C. et al. (2017). Model predictive control for cost-minimizing
  EV charging with rooftop PV. *Applied Energy* 196: 165–177.
