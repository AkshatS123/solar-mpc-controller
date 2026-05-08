# Launch assets

> Day-0 publishing assets for the `solar-mpc-controller` launch.
> Edit voice and personalize before sending. Order matches the strategic
> plan's launch arc.

---

## Twitter thread (8 tweets, ~280 chars each)

**1/**
Most EV chargers do the wrong thing when the sun's out.

I built the right thing — open-source, MIT, runs in cvxpy. Repo + writeup ↓

🔗 https://github.com/AkshatS123/solar-mpc-controller

**2/**
The setup: rooftop solar + EV + TOU pricing. Naive automations charge "when there's solar excess." That's a greedy heuristic, not optimization.

The actual problem is receding-horizon model-predictive control:

```
min Σ price[τ] · max(0, load + charge − solar) · Δt
s.t. charge ∈ [0, amp_max], SoC[deadline] ≥ target, slew limits
```

**3/**
Three controllers on a 2-day synthetic trace, 60 kWh / 240 V / 32 A:

[image: 02_comparison.png]

MPC: $6.81. Greedy: $7.67 (+13%). TOU: $10.82 (+59%).

The MPC's amperage trajectory hugs the solar curve. Greedy bangs full-on. TOU square-waves through off-peak.

**4/**
Then I added forecast noise — log-normal multiplicative on solar, Gaussian on load. 50 realizations, tight deadline (0.25→0.90 SoC by noon).

[image: 03_robustness.png]

**5/**
The headline: **Greedy is the cheapest controller, and it misses the deadline 100% of the time.** Cheapness is an artifact of failing the job.

MPC: 0% miss. TOU: 0% miss. Greedy: 0/50.

**6/**
I built a RobustMPC variant (scenario-based, K=10) expecting it to dominate. It didn't — same reliability as nominal, $0.03 more expensive.

Reason: SoC dynamics are deterministic given the action, so the deadline constraint is deterministic. Robust formulations only differ on cost variance.

**7/**
Where robust *would* dominate:
- tight battery/charger margins (commercial fleets)
- chance-constrained deadlines (compliance SLAs)
- asymmetric forecast bias (cloudy-day systematic error)
- multi-stage stochastic programs with recourse

Shipped the variants anyway. Just not the headline.

**8/**
Repo: https://github.com/AkshatS123/solar-mpc-controller
Algorithm spec: docs/algorithm.md
Walkthrough: notebooks/01_walkthrough.md
Pairs with: github.com/AkshatS123/solar-tsfm-bench

Working on residential energy controls, TSFMs for solar, or imitation learning for control? DM me.

---

## LinkedIn post (~600 words)

> **Why most EV chargers leave money on the table — and what to do about it**

If you have rooftop solar and an EV, you have a small optimization problem in your driveway every day.

Your panels produce 6 kW at noon. Your house draws ~1 kW. Your EV takes 7.7 kW at full charge. Your utility charges $0.18/kWh off-peak and $0.45/kWh during 4–9 pm peak.

The naive answer: "charge when there's solar excess." Most consumer products — Tesla mobile app, Wallbox Eco mode, Enphase Encharge — do exactly that. It's a greedy heuristic, not optimization.

I just open-sourced the right answer: **model-predictive control** for residential EV+solar charging. cvxpy + CLARABEL, ~500 LOC, MIT licensed.

📦 github.com/AkshatS123/solar-mpc-controller

**The result**

On a 2-day synthetic trace (60 kWh battery, 32 A charger, TOU pricing), the gap is consistent and meaningful:

→ MPC: $6.81
→ Greedy (charge on solar excess): $7.67 (+13%)
→ TOU (charge on off-peak prices): $10.82 (+59%)

The MPC's amperage trajectory *hugs* the solar curve — back off when peak prices kick in, opportunistically top up using next day's solar. That's the look-ahead earning its keep.

**The honest finding**

I expected my robust-MPC variant (10 forecast scenarios, optimize over expected cost) to dominate the nominal MPC under noisy forecasts. It didn't.

Reason: SoC dynamics depend only on the charging action (the decision variable), not on uncertain solar/load. So the deadline constraint is deterministic, and robust formulations only differentiate on cost variance — which is small when the problem has margin.

Robust formulations earn their complexity in:
• Tight battery/charger margins (commercial fleets)
• Chance-constrained deadlines (compliance SLAs)
• Asymmetric forecast bias (cloudy-day systematic error)
• Multi-stage stochastic programs with recourse

I shipped the robust + CVaR + mixed-integer variants anyway as documented opt-in extensions. They're correct; they're just not the headline story for residential EV charging.

**What's also surprising**

Greedy is the cheapest controller in the noisy comparison ($8.65) — and **misses the deadline 100% of the time**. The cheapness is an artifact of failing at the actual job. This is the trap most consumer EV automations fall into.

**Why I wrote it**

I'm building Tesphase (tesphase.co), a home-energy platform that charges EVs from rooftop solar. The v2 charging algorithm is MPC; this repo is the public reference implementation behind it. Pairs with `solar-tsfm-bench`, a benchmark for time-series foundation models on residential solar forecasting.

If you're working on residential energy controls, time-series forecasting for solar, or imitation learning for control policies — let's talk.

🔗 github.com/AkshatS123/solar-mpc-controller

#energy #optimization #cleantech #opensource #ml #controls

---

## Cold-email template (~150 words)

> **Subject:** open-source MPC for residential EV+solar — would value your eyes

Hi [Name],

Saw your [paper / talk / project on X] — really liked the [specific insight to compliment, not generic].

I just open-sourced a small reference implementation of model-predictive control for residential EV charging from rooftop solar: a convex-QP formulation in cvxpy, scenario-based robust variant, and a stochastic simulator that runs the comparison against greedy + TOU baselines under noisy forecasts.

What I think might be interesting to you:

- The honest finding that nominal MPC is already robust under unbiased noise (deterministic SoC dynamics) — I expected RobustMPC to dominate, it didn't, and the writeup is upfront about why.
- The pairing with `solar-tsfm-bench` — TSFM forecasts → robust MPC, two repos one story.

Repo: https://github.com/AkshatS123/solar-mpc-controller
Writeup: [link to akshatsharma.blog post once published]

Would value any thoughts — what to fix, what's missing, what I'm wrong about.

Thanks,
Akshat

---

## Outreach target list (suggested starting points)

> **You decide who actually fits.** These are categories to look at, not
> a finished list. Aim for 3–5 emails on Day 1.

| Category | Who to look at |
|---|---|
| Stephen Boyd's group alumni | recent PhDs working on energy, control, or convex optimization in industry |
| Applied Energy MPC residential authors | first-author of any 2024–2026 paper on home-EV MPC |
| IEEE TSG building/EV controls | recent special issue authors |
| YC energy founders | Y22-onward batches with home-energy startups |
| Tesphase ecosystem | Toby Corey's network, ex-SolarCity engineers |

---

## Day-0 checklist (engineer-only steps; everything else is the user)

- [ ] All four figures regenerate cleanly: `uv run python scripts/plot_walkthrough.py`
- [ ] All tests green + ruff clean
- [ ] README + walkthrough + algorithm spec final pass
- [ ] `git tag v0.1.0 && git push --tags` (user — auth-gated)
- [ ] GitHub release created with these notes (user)
- [ ] Blog post published on akshatsharma.blog (user)
- [ ] Twitter thread (user)
- [ ] LinkedIn post (user)
- [ ] 3–5 cold emails sent (user)
- [ ] Update `Accomplishments.md` in vault (user)
- [ ] +14 days: backfill star/fork numbers in resume bullet (user)
