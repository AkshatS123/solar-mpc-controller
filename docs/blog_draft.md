# An open-source MPC for charging your EV from solar

> Draft for [akshatsharma.blog](https://akshatsharma.blog). ~1500 words.
> Edit in your voice; this is a structural skeleton with placeholders.

---

## The problem

If you have rooftop solar and an EV, you have a small optimization problem
sitting in your driveway every day.

Your panels produce 6 kW at noon. Your house draws maybe 1 kW. Your EV
takes 7.7 kW at full charge (240 V × 32 A). Your utility charges $0.18/kWh
off-peak and $0.45/kWh during the 4–9 pm peak.

The naïve answer is "charge when there's solar excess." Most consumer
products do exactly this. The Tesla mobile app, Wallbox Eco mode, Enphase
Encharge — they all greedy-match charging to instantaneous solar surplus.

I'll show you why that's wrong, what's actually right, and the open-source
implementation: [`solar-mpc-controller`](https://github.com/AkshatS123/solar-mpc-controller).

## What the optimal controller looks like

The right tool here is **model predictive control** (MPC). At each control
tick (every 15 minutes, say), you solve:

> Given my current battery state, the next 6 hours of solar forecast,
> load forecast, and grid price, what amperage trajectory minimizes my
> total cost while still hitting the SoC I need by my deadline?

You apply the first action of the plan, throw the rest away, and re-solve
on the next tick. This is the "receding horizon" property — wrong forecasts
get corrected immediately because the optimization re-runs.

Mathematically:

```
minimize   Σ_τ price[τ] · max(0, load[τ] + charge[τ] − solar[τ]) · Δt
         + λ_smooth · Σ_τ (charge[τ+1] − charge[τ])²
         − λ_soc · SoC[H−1]

subject to 0 ≤ charge[τ] ≤ amp_max
           |charge[τ+1] − charge[τ]| ≤ Δamp_max
           SoC propagates from charge
           SoC[deadline] ≥ SoC_target
```

That's a convex quadratic program. cvxpy + CLARABEL solves it in
milliseconds. The full spec is in [`docs/algorithm.md`](https://github.com/AkshatS123/solar-mpc-controller/blob/main/docs/algorithm.md).

## What the comparison looks like

Three controllers on a 2-day synthetic trace (60 kWh battery, 32 A
charger, TOU pricing):

![comparison](https://raw.githubusercontent.com/AkshatS123/solar-mpc-controller/main/notebooks/figures/02_comparison.png)

- **MPC** (blue) ramps charging into the noon solar peak, parks at 0.8
  SoC right at the target, opportunistically tops up using day 2's
  solar. **Cost: $6.81.**
- **Greedy** (orange) waits for solar excess, then ramps to 32 A. Hits
  100 % SoC and sits there. **Cost: $7.67 — 13 % more than MPC.**
- **TOU** (purple) ignores solar entirely, charges hard during off-peak.
  **Cost: $10.82 — 59 % more than MPC.**

The MPC's amperage panel (bottom of the figure) is what makes this
visually clear: it *hugs* the solar curve. Greedy bangs full-on. TOU
squares-waves through off-peak.

## Then I tried it with bad forecasts

Perfect forecasts are a lie. Real solar predictions are off by 20–40 %
on cloudy days. So I added forecast noise — log-normal multiplicative
on solar, additive Gaussian on load — and ran 50 noisy realizations of
the same trace with a tighter deadline (SoC 0.25 → 0.90 by noon):

![reliability](https://raw.githubusercontent.com/AkshatS123/solar-mpc-controller/main/notebooks/figures/03_robustness.png)

The headline:

> **Greedy is the cheapest controller — and misses the deadline 100 %
> of the time.**

It charges when solar exceeds load, full stop. On a noisy realization
where solar comes in below forecast, Greedy under-charges and never
catches up. The cheapness is an artifact of failing the actual job.

MPC and TOU both reliably hit the deadline. MPC pays $9.91; TOU pays
$12.52 (a 26 % premium for ignoring solar).

## The thing I expected to find but didn't

I built a `RobustMPCController` that takes K=10 forecast scenarios
instead of one point forecast and minimizes expected cost across all of
them. I expected it to dominate nominal MPC under noise.

It didn't. RobustMPC came in at $9.94 — three cents more expensive than
nominal MPC, identical reliability.

The reason is a property of the formulation: SoC propagation depends only
on the action (the amperage trajectory), not on the noisy solar/load. So
the deadline constraint is *deterministic*. Robust formulations only
differentiate on **cost variance**, which is small here because the
problem has plenty of margin.

I added a CVaR-weighted variant (Rockafellar–Uryasev form) that explicitly
trades mean cost for tail-cost reduction. It moved cost by $0.06.

Honest finding: **on this problem, nominal MPC is already robust.**
Robust extensions earn their complexity in different regimes:

- Tight battery / charger margins (commercial fleets, grid-edge)
- Chance-constrained deadlines (compliance-mandated SLAs)
- Asymmetric or biased forecast errors (cloudy-day systematic bias)
- Multi-stage stochastic programs with recourse (long-horizon planning)

I shipped the robust + CVaR + mixed-integer variants anyway — they're
correct extensions and the code documents the contract. They're just not
the headline story for residential EV charging.

## Why I wrote this

I'm building [Tesphase](https://tesphase.co), a home-energy product that
charges EVs from rooftop solar. The v2 charging algorithm is MPC. This
repo is the public reference implementation — a clean, citable place to
point when someone asks "how does Tesphase decide when to charge?"

It also pairs with [`solar-tsfm-bench`](https://github.com/AkshatS123/solar-tsfm-bench),
a benchmark for time-series foundation models on residential solar
forecasting. The benchmark answers "how good is the forecast?" The
controller answers "given the forecast, what should you do?" The
imitation-learning follow-up will distill this MPC into a small policy
network for cheaper inference.

## What's not in the model

- **Demand charges** ($/kW per month). PG&E EV2-A residential customers
  don't see them; commercial customers care a lot more.
- **Battery degradation cost.** Charging hard always looks free in this
  cost model. Real economic models include $/cycle.
- **Multi-vehicle coordination.** Two-EV households are coming; this
  controller is single-asset.
- **Grid-services revenue** (V2G, demand response). Bigger problem,
  later.

## Try it

```bash
git clone https://github.com/AkshatS123/solar-mpc-controller
cd solar-mpc-controller
uv sync
uv run pytest -q
uv run python scripts/run_demo.py
uv run python scripts/plot_walkthrough.py
uv run solar-mpc run --controller mpc
```

MIT licensed. Issues / PRs welcome.

If you're working on residential energy controls, on TSFMs for solar, or
on imitation learning for control policies, send me a note —
[s.akshat@gmail.com](mailto:s.akshat@gmail.com).

---

## Author note (delete before publishing)

Tone: technical, honest, no marketing-speak. The "honest finding"
section is the differentiator vs every other MPC blog post.

Recommended cuts if length is a problem:
- "Why I wrote this" → drop to one sentence linking Tesphase
- "What's not in the model" → drop entirely if word count is tight

Things to verify before publishing:
- Confirm PG&E EV2-A is the right tariff to name (or pick SCE TOU-D-PRIME
  if SoCal-targeted)
- Replace github.com URLs with final permalinks if you fork-rename
- Add the cover image (recommend 02_comparison.png cropped to top-third)

Hook variants to A/B:
1. "If you have rooftop solar and an EV, you have a small optimization problem sitting in your driveway every day."
2. "Most EV chargers do the wrong thing when the sun's out. Here's the right thing."
3. "I expected my robust controller to beat the nominal one. It didn't. Here's why that's actually good."

Twitter thread arc (8 tweets):
1. Hook — most EV chargers greedy-match solar, that's wrong
2. The math — MPC in 6 lines
3. Figure 02 — the perfect-forecast comparison
4. Then I added forecast noise (lead-in)
5. Figure 03 — Greedy 100% deadline-miss
6. The unexpected: RobustMPC barely beat nominal MPC, and why
7. Where robust *would* win — tight margins, chance constraints, fleets
8. Repo link + ask
