#!/usr/bin/env python3
"""
Identifying the world from outcomes: a Bayesian bettor in the fixed-world lottery.

decide_kelly_unknown_world.py prices ignorance of the world: the best fixed rule (mixworld)
gives up 0.04-0.06 bits per round against the world-knowing oracle, forever. This script shows
that in the fixed-world version the price need not be paid per round. A bettor who starts from
the uniform prior over the K candidate worlds, bets the posterior-weighted mixture of the world
formulas (her first bet is mixworld), and updates the posterior with each outcome's likelihood,
has final wealth equal to the plain average of the wealths of the K single-world bettors.
Hence on every sequence she ends at most log2(K) bits behind the bettor who knew the true
world from the start -- the classical mixture bound of universal prediction (Cesa-Bianchi and
Lugosi 2006, ch. 9; Cover and Thomas 2006, ch. 6) -- and her long-run growth rate equals the
oracle's. Ignorance of the world costs a one-time identification fee, not a rate.

Reported per group: the bettor's cumulative shortfall against the world-knowing oracle (bits;
mean over true worlds and seeds, and the worst world-seed pair) at several horizons, next to
the log2(K) bound, plus the round by which half of the final shortfall has been paid.

Reuses the worlds, report distributions and seeds of decide_kelly_unknown_world.py unchanged.
"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decide_kelly_unknown_world import (WORLDS_A, WORLDS_B, TRUTHS_A, TRUTHS_B,
                                        rounds_A, rounds_B)

T = 100_000                                   # rounds per fixed-world series
SEEDS = (11, 12, 13, 14, 15)                  # several series per true world
CHECKPOINTS = (10, 30, 100, 1000, 10_000, T)


def clip(x):
    return min(max(x, 1e-12), 1.0 - 1e-12)


def bayes_series(truths, world, T, seed, rounds_fn):
    """One fixed-world series. Returns the cumulative shortfall (bits) of the
    posterior-mixture bettor against the oracle who bets the true world's formula,
    recorded after every round."""
    names = list(truths)
    logw = {w: 0.0 for w in names}            # unnormalized log posterior weights
    shortfall, behind = [], 0.0
    for p_true, out, x, y in rounds_fn(world, T, seed):
        fs = {w: clip(truths[w](x, y)) for w in names}
        m = max(logw.values())
        ws = {w: math.exp(logw[w] - m) for w in names}
        q = clip(sum(ws[w] * fs[w] for w in names) / sum(ws.values()))
        p = clip(p_true)
        behind += (math.log2(p if out else 1 - p)
                   - math.log2(q if out else 1 - q))
        shortfall.append(behind)
        for w in names:
            logw[w] += math.log(fs[w] if out else 1 - fs[w])
    return shortfall


def report(label, worlds, truths, rounds_fn, seed0):
    K = len(worlds)
    bound = math.log2(K)
    print(f"--- {label}: K = {K} worlds, mixture bound log2(K) = {bound:.3f} bits ---")
    curves = []
    for i, w in enumerate(worlds):
        for s in SEEDS:
            # seed0 + i reproduces the unknown-world pairing for the first seed
            curves.append(bayes_series(truths, w, T, seed0 + i + 1000 * (s - SEEDS[0]),
                                       rounds_fn))
    n = len(curves)
    for t in CHECKPOINTS:
        vals = [c[t - 1] for c in curves]
        print(f"  after {t:>6d} rounds: shortfall vs oracle"
              f"  mean {sum(vals) / n:6.3f} bits   worst {max(vals):6.3f}")
    finals = [c[-1] for c in curves]
    mean_curve = [sum(c[t] for c in curves) / n for t in range(T)]
    half = next(t + 1 for t in range(T) if mean_curve[t] >= 0.5 * mean_curve[-1])
    print(f"  half of the final mean shortfall is paid within the first {half} rounds")
    ok = max(finals) <= bound + 1e-9
    print(f"  bound respected on every series: {'yes' if ok else 'NO'}")
    print(f"  growth-rate cost at the {T}-round horizon:"
          f" {sum(finals) / n / T:.6f} bits/round\n")


def main():
    print("=== identifying the world by betting (fixed-world lottery, uniform prior) ===")
    print("(the bettor bets the posterior mixture of the world formulas; her first bet")
    print(" is mixworld; the oracle bets the true world's formula from round one)\n")
    report("group A: both numbers support the event",
           WORLDS_A, TRUTHS_A, rounds_A, seed0=11)
    report("group B: for-number and against-number",
           WORLDS_B, TRUTHS_B, rounds_B, seed0=31)


if __name__ == "__main__":
    main()
