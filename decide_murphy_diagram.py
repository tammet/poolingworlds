#!/usr/bin/env python3
"""
Murphy diagram: the cost-loss expense of each pooling rule at every stake level at once.

The keeper of decide_cost_loss.py takes a precaution costing c (as a share of the preventable
loss, L = 1) when the pooled number exceeds c. Spending is counted from the perfect-foresight
zero point (know each outcome in advance, pay c on exactly the event rounds): a wasted
precaution burns c, an unprotected event costs 1 where foresight would have paid c, so 1-c
extra. This avoidable expense at one c is one number; the Murphy diagram (Ehm, Gneiting,
Jordan, Krueger, JRSS B 2016) plots it against all c in (0,1), one curve per rule; lower is
better, and the oracle (the keeper handed the true probability) is the floor. Two worlds: the two-habitats world
(truth = average; reports drawn as two independent uniforms so the curves are smooth --
the fixed report menu of decide_kelly_betting.world_B gives the same story as step functions)
and the two-experts world of decide_kelly_betting.py (truth = upco).

What the picture shows:
  * the oracle curve (keeper handed the true probability) is the floor at every c;
  * the situation-matched rule lies on the floor along the whole axis;
  * the four rules of Proposition 1 (avg, geo, upco, mycin) pay the same at c = 1/2 --
    their curves meet in one point -- and separate toward the cheap and expensive ends;
  * noisy-or does not pass through that point: it can choose a different side at even stakes.

The Schervish 1989 identity ("Kelly growth = the aggregate of the cost-loss expenses over all
stake levels"): weighting the expense curve by dc/(c(1-c)) and integrating reproduces the
log score, so the area between a rule's curve and the oracle's, under that weight, is the
Kelly growth given up by the rule. The script checks this numerically: the weighted integral
of the expense gap matches the betting gap of decide_kelly_betting.py on the same rounds.

Output: murphy_diagram.pdf/.png written into this folder (and into ../paper_pooling/ when
that directory exists, keeping the paper's copy of the figure current) + a numeric table +
the integral check. Needs numpy and matplotlib; the rest of the suite is pure Python.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
import numpy as np
import poolib
from decide_kelly_betting import world_A, growth_table, std_pools

T = 100_000


def world_mixture_uniform(T=T, seed=1):
    """Two-habitats world with continuous reports: p1, p2 ~ U(0.15, 0.85), one habitat real
    (coin flip), animal drawn at the real habitat's rate. Truth = (p1+p2)/2."""
    random.seed(seed)
    rounds = []
    for _ in range(T):
        p1, p2 = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        rate = p1 if random.random() < 0.5 else p2
        rounds.append(((p1 + p2) / 2, random.random() < rate, p1, p2))
    return rounds

RULES = ["avg", "geo", "upco", "mycin", "noisyor"]   # plotted rules, palette slot order
FULL = {"avg": "average", "geo": "geometric", "upco": "upco", "mycin": "mycin",
        "noisyor": "noisy-or", "oracle": "oracle"}


def curves(rounds, grid):
    """Mean avoidable expense at each cost ratio in grid, per rule.
    Elementary loss vs perfect foresight: (1-c) if the event happens unprotected
    (q <= c), c if the precaution was wasted (q > c, no event)."""
    p_true = np.array([r[0] for r in rounds])
    y = np.array([r[1] for r in rounds])
    p1 = np.array([r[2] for r in rounds])
    p2 = np.array([r[3] for r in rounds])
    qs = {"oracle": p_true}
    fns = {"avg": poolib.avg, "geo": poolib.geo, "upco": poolib.upco,
           "mycin": poolib.mycin, "noisyor": poolib.noisy_or}
    for name in RULES:
        qs[name] = np.array([fns[name](a, b) for a, b in zip(p1, p2)])
    out = {}
    for name, q in qs.items():
        loss = np.empty(len(grid))
        for i, c in enumerate(grid):
            act = q > c
            loss[i] = np.mean(np.where(y, np.where(act, 0.0, 1.0 - c),
                                          np.where(act, c, 0.0)))
        out[name] = loss
    return out


def schervish_check(rounds, label):
    """Integral of the expense gap with weight dc/(c(1-c)) vs the Kelly gap in bits."""
    grid = np.arange(0.0005, 1.0, 0.001)          # midpoint rule
    cv = curves(rounds, grid)
    w = 1.0 / (grid * (1.0 - grid))
    rates = growth_table(rounds, std_pools(0.5), prior=0.5)
    print(f"  {label}: weighted integral of the expense gap vs Kelly gap (bits/round)")
    for name in RULES:
        integral = np.sum((cv[name] - cv["oracle"]) * w) * 0.001 / math.log(2)
        kelly = rates["oracle"] - rates[name]
        print(f"    {name:8s} integral {integral:.4f}   betting gap {kelly:.4f}"
              f"   difference {abs(integral - kelly):.4f}")
    return cv


def table(rounds, label):
    print(f"  {label}: avoidable expense at selected cost ratios")
    cs = (0.2, 0.5, 0.8)
    cv = curves(rounds, np.array(cs))
    print(f"    {'rule':8s} {'c=0.2':>8s} {'c=0.5':>8s} {'c=0.8':>8s}")
    for name in ["oracle"] + RULES:
        cells = "  ".join(f"{cv[name][i]:8.4f}" for i in range(len(cs)))
        print(f"    {name:8s} {cells}")
    spread = max(cv[n][1] for n in RULES[:4]) - min(cv[n][1] for n in RULES[:4])
    print(f"    -> spread of avg/geo/upco/mycin at c=0.5: {spread:.6f}"
          f" (Proposition 1: identical decisions, identical expense)\n")


def plot(cv_B, cv_A, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted = "#ffffff", "#0b0b0b", "#52514e", "#898781"
    grid_c, axis_c = "#e1e0d9", "#c3c2b7"
    series = {"avg": "#2a78d6", "geo": "#1baf7a", "upco": "#eda100",
              "mycin": "#008300", "noisyor": "#4a3aa7"}
    dashes = {"avg": (), "geo": (), "upco": (), "mycin": (), "noisyor": (5, 2, 1, 2)}

    grid = np.arange(0.0005, 1.0, 0.001)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharey=True, dpi=200,
                             facecolor=surface)
    panels = [(axes[0], cv_B, "two habitats (truth = average)"),
              (axes[1], cv_A, "two experts (truth = upco)")]
    for ax, cv, title in panels:
        ax.set_facecolor(surface)
        ax.axvline(0.5, color=grid_c, lw=0.8, zorder=0)
        ax.grid(axis="y", color=grid_c, lw=0.6, zorder=0)
        ax.plot(grid, cv["oracle"], color=ink, lw=2.6, ls=(0, (4, 2)), alpha=0.55,
                label="oracle (floor)", zorder=2)
        for name in RULES:
            ax.plot(grid, cv[name], color=series[name], lw=1.9,
                    dashes=dashes[name] or (None, None),
                    label=FULL[name], zorder=3)
        ax.set_title(title, fontsize=11, color=ink)
        ax.set_xlabel("precaution price c  (share of the preventable loss)",
                      fontsize=10, color=ink2)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis_c)
    axes[0].set_ylabel("avoidable expense per round\n(lower is better)",
                       fontsize=10, color=ink2)

    # direct labels in clear zones (positions hand-placed per panel)
    def mark(ax, text, x, y, color, ha="center"):
        ax.text(x, y, text, fontsize=8.5, color=color, ha=ha)
    mark(axes[0], "noisy-or", 0.74, 0.222, series["noisyor"])
    mark(axes[0], "upco, mycin", 0.16, 0.100, series["upco"], ha="right")
    mark(axes[0], "average (= oracle)", 0.42, 0.128, series["avg"])
    mark(axes[1], "noisy-or", 0.42, 0.163, series["noisyor"])
    mark(axes[1], "average", 0.65, 0.106, series["avg"])
    mark(axes[1], "upco (= oracle)", 0.50, 0.077, series["upco"])

    axes[1].legend(loc="upper right", fontsize=9, frameon=False,
                   labelcolor=ink2, handlelength=2.2)
    fig.tight_layout()
    for d in outdir:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(d, f"murphy_diagram.{ext}"),
                        facecolor=surface, bbox_inches="tight")
        print(f"  figure written to {d}/murphy_diagram.pdf and .png")


def main():
    print("=== Murphy diagram: cost-loss expense at every stake level, per pooling rule ===\n")
    rounds_B = world_mixture_uniform(T=T, seed=1)
    rounds_A = world_A(T=T, seed=1, prior=0.5)
    cv_B = schervish_check(rounds_B, "two-habitats world")
    print()
    cv_A = schervish_check(rounds_A, "two-experts world")
    print()
    table(rounds_B, "two-habitats world")
    table(rounds_A, "two-experts world")
    here = os.path.dirname(os.path.abspath(__file__))
    outdirs = [here]
    paper_dir = os.path.normpath(os.path.join(here, "..", "paper_pooling"))
    if os.path.isdir(paper_dir):
        outdirs.append(paper_dir)
    plot(cv_B, cv_A, outdirs)


if __name__ == "__main__":
    main()
