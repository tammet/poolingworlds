#!/usr/bin/env python3
"""
Generate the illustration figures for the pooling paper (except the Murphy diagram,
which is produced by decide_murphy_diagram.py):

  fig_rules_probability  -- the probability-scale pools side by side: f(x,y) vs x
                            for fixed y (two panels, y = 0.3 and y = 0.7)
  fig_rules_confidence   -- the confidence-scale combinators side by side
                            (two panels, y = 0.25 and y = 0.75)
  fig_calibration        -- part-one illustration: observed frequency vs pooled value
                            per rule in the experts world and the habitats world;
                            the situation-matched rule lies on the diagonal
  fig_wealth             -- part-two illustration: cumulative log2 wealth of one
                            bettor per rule over 500 rounds in the same two worlds
  fig_alpha              -- growth given up vs oracle across the extremizing
                            exponent alpha, four evidence regimes

Needs numpy + matplotlib. Writes .pdf and .png into this directory.
"""
import os, sys, math, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import poolib
from decide_kelly_betting import world_A, world_B
from decide_murphy_diagram import world_mixture_uniform
from decide_extremized_weighted_logodds import (rounds_private, rounds_shared,
                                                rounds_halfshared)

SURFACE, INK, INK2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C = "#e1e0d9", "#c3c2b7"
COLOR = {                                  # one fixed color per rule, all figures
    "avg":     "#2a78d6",
    "geo":     "#1baf7a",
    "upco":    "#eda100",
    "mycin":   "#008300",
    "noisyor": "#4a3aa7",
    "max":     "#e34948",
    "problog": "#eb6834",
}
LABEL = {"avg": "average", "geo": "geometric", "upco": "upco", "mycin": "mycin",
         "noisyor": "noisy-or", "max": "max", "problog": "problog"}
DASH = {"noisyor": (5, 2, 1, 2), "max": (4, 2)}     # help grayscale printing


def style(ax, xlabel, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID_C, lw=0.6, zorder=0)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_C)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=INK2)


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"),
                    facecolor=SURFACE, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  written {name}.pdf/.png")


def curveplot(rules, ys, name, note):
    xs = np.arange(0.0, 1.0001, 0.005)
    fig, axes = plt.subplots(1, len(ys), figsize=(9.0, 3.4), sharey=True,
                             dpi=200, facecolor=SURFACE)
    for ax, y in zip(axes, ys):
        for r in rules:
            vals = [getattr(poolib, {"noisyor": "noisy_or", "max": "mx",
                                     "avg": "avg", "geo": "geo", "upco": "upco",
                                     "mycin": "mycin", "problog": "problog"}[r])(x, y)
                    for x in xs]
            ax.plot(xs, vals, color=COLOR[r], lw=1.9,
                    dashes=DASH.get(r, (None, None)), label=LABEL[r], zorder=3)
        ax.set_title(f"second input fixed at {y:g}", fontsize=11, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        style(ax, "first input x")
    axes[0].set_ylabel(f"pooled value ({note})", fontsize=10, color=INK2)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False,
                   labelcolor=INK2, handlelength=2.4)
    save(fig, name)


def fig_calibration():
    """Observed frequency of the event vs the pooled value, 25 equal-count bins."""
    worlds = [("two experts (truth = upco)", world_A(T=200_000, seed=1, prior=0.5)),
              ("two habitats (truth = average)", world_mixture_uniform(T=200_000, seed=1))]
    rules = ["avg", "geo", "upco", "mycin", "noisyor"]
    fns = {"avg": poolib.avg, "geo": poolib.geo, "upco": poolib.upco,
           "mycin": poolib.mycin, "noisyor": poolib.noisy_or}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharey=True,
                             dpi=200, facecolor=SURFACE)
    for ax, (title, rounds) in zip(axes, worlds):
        y = np.array([r[1] for r in rounds], dtype=float)
        p1 = np.array([r[2] for r in rounds]); p2 = np.array([r[3] for r in rounds])
        ax.plot([0, 1], [0, 1], color=INK, lw=2.0, ls=(0, (4, 2)), alpha=0.45,
                label="calibrated (diagonal)", zorder=2)
        for r in rules:
            q = np.array([fns[r](a, b) for a, b in zip(p1, p2)])
            order = np.argsort(q)
            qs, ys_ = q[order], y[order]
            nb = 25
            edges = np.linspace(0, len(q), nb + 1).astype(int)
            mq = [qs[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]
            mf = [ys_[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(mq, mf, color=COLOR[r], lw=1.9,
                    dashes=DASH.get(r, (None, None)), label=LABEL[r], zorder=3)
        ax.set_title(title, fontsize=11, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        style(ax, "pooled value q")
    axes[0].set_ylabel("observed frequency of the event", fontsize=10, color=INK2)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False,
                   labelcolor=INK2, handlelength=2.4)
    save(fig, "fig_calibration")


def fig_wealth(T=500, seed=9):
    """One bettor per rule, cumulative log2 wealth over T rounds, even odds."""
    worlds = [("two experts (truth = upco)", world_A(T=T, seed=seed, prior=0.5)),
              ("two habitats (truth = average)", world_B(T=T, seed=seed))]
    rules = ["avg", "geo", "upco", "noisyor"]
    fns = {"avg": poolib.avg, "geo": poolib.geo, "upco": poolib.upco,
           "noisyor": poolib.noisy_or}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharex=True,
                             dpi=200, facecolor=SURFACE)
    for ax, (title, rounds) in zip(axes, worlds):
        series = {r: [0.0] for r in rules}
        series["oracle"] = [0.0]
        for p_true, out, p1, p2 in rounds:
            for r in rules + ["oracle"]:
                q = p_true if r == "oracle" else fns[r](p1, p2)
                q = min(max(q, 1e-6), 1 - 1e-6)
                q_out = q if out else 1 - q
                series[r].append(series[r][-1] + 1 + math.log2(q_out))
        ax.axhline(0, color=GRID_C, lw=0.8, zorder=0)
        ax.plot(series["oracle"], color=INK, lw=2.2, ls=(0, (4, 2)), alpha=0.5,
                label="oracle", zorder=2)
        for r in rules:
            ax.plot(series[r], color=COLOR[r], lw=1.7,
                    dashes=DASH.get(r, (None, None)), label=LABEL[r], zorder=3)
        ax.set_title(title, fontsize=11, color=INK)
        style(ax, "round")
    axes[0].set_ylabel("wealth (log2 of the multiple;\n0 = starting wealth)",
                       fontsize=10, color=INK2)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False,
                   labelcolor=INK2, handlelength=2.4)
    save(fig, "fig_wealth")


def fig_alpha(T=100_000):
    """Growth given up vs the oracle across the symmetric exponent alpha."""
    regimes = [
        ("calibrated, private evidence (best: 1)", rounds_private(T=T, k=1.0), "#2a78d6", ()),
        ("fully shared evidence (best: 1/2)",      rounds_shared(T=T),          "#1baf7a", ()),
        ("timid reports, k = 1/2 (best: 2)",       rounds_private(T=T, k=0.5),  "#eda100", ()),
        ("half-shared evidence (no exact alpha)",  rounds_halfshared(T=T),      "#e34948", (4, 2)),
    ]
    alphas = np.arange(0.4, 2.6001, 0.05)
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200, facecolor=SURFACE)
    for label, rounds, color, dash in regimes:
        p_true = np.array([r[0] for r in rounds]); out = np.array([r[1] for r in rounds])
        p1 = np.clip(np.array([r[2] for r in rounds]), 1e-12, 1 - 1e-12)
        p2 = np.clip(np.array([r[3] for r in rounds]), 1e-12, 1 - 1e-12)
        L = np.log(p1 / (1 - p1)) + np.log(p2 / (1 - p2))
        pt = np.clip(p_true, 1e-9, 1 - 1e-9)
        g_oracle = np.mean(np.log2(np.where(out, pt, 1 - pt))) + 1
        gaps = []
        for a in alphas:
            q = np.clip(1 / (1 + np.exp(-a * L)), 1e-9, 1 - 1e-9)
            g = np.mean(np.log2(np.where(out, q, 1 - q))) + 1
            gaps.append(g_oracle - g)
        gaps = np.array(gaps)
        ax.plot(alphas, gaps, color=color, lw=1.9, dashes=dash or (None, None),
                label=label, zorder=3)
        i = int(np.argmin(gaps))
        ax.plot([alphas[i]], [gaps[i]], "o", ms=4, color=color, zorder=4)
    ax.axhline(0, color=AXIS_C, lw=0.8, zorder=1)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(bottom=-0.005)
    style(ax, "exponent alpha (both weights equal to alpha)",
          "growth given up vs oracle (bits/round)")
    ax.legend(loc="upper right", fontsize=9, frameon=False,
              labelcolor=INK2, handlelength=2.4)
    save(fig, "fig_alpha")


def main():
    print("generating paper figures ...")
    curveplot(["avg", "geo", "upco", "max", "noisyor"], [0.3, 0.7],
              "fig_rules_probability", "probability scale")
    curveplot(["mycin", "problog", "upco"], [0.25, 0.75],
              "fig_rules_confidence", "confidence scale")
    fig_calibration()
    fig_wealth()
    fig_alpha()


if __name__ == "__main__":
    main()
