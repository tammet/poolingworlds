#!/usr/bin/env python3
"""
Using the pooled number to ACT: a keeper who must decide whether to take a costly precaution.

Story. A wildlife keeper hears the two observers' reports about whether the incoming animal is
a bird of prey. Taking a precaution (covering the aviary) costs C; doing nothing risks damage
L = 1 if the animal really is a predator. The rational rule: pool the reports into q and act
iff q > C/L. This is the classic cost-loss decision model used to price weather forecasts
(Murphy 1977; Richardson 2000): the keeper's average expense, over many animals and a range of
cost ratios C/L, measures how useful her pooling rule is in practice. The true probability
minimizes the expense at EVERY cost ratio simultaneously; a wrong pool pays extra at the ratios
where it lands on the wrong side of the threshold.

Part 0 first proves the fact that doomed the older "count correct guesses" experiments
(gpt_v7 / suva_decision_accuracy / the identical-accuracy artifact in gpt_v10): at the even
threshold C/L = 1/2, avg, geometric, upco and mycin all make the SAME decision -- each of them
is above 1/2 exactly when p1 + p2 >= 1. So counting correct guesses at even stakes cannot
distinguish these four pools at all; differences only appear at uneven stakes (this file) or
through bet sizing (decide_kelly_betting.py). Noisy-or and max do decide differently even at
1/2 (they are not opinion pools but evidence accumulators).

Worlds as in decide_kelly_betting.py:
  A  two separate experts, prior 1/2: true probability = upco  -> upco-keeper pays least.
  B  mixture world: true probability = average                 -> avg-keeper pays least.

Reported: average expense per animal (lower is better) for each pool at each cost ratio, with
the oracle (true probability) as the floor. Watch the C/L = 0.5 column: avg, geo, upco and mycin
are identical there, and split apart at 0.2 and 0.8.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib
from decide_kelly_betting import world_A, world_B, std_pools


def part0_threshold_half_equivalence(step=0.01):
    print("=== Part 0: at threshold 1/2, avg / geo / upco / mycin make the SAME decision ===")
    print("(each is > 1/2 exactly when p1 + p2 > 1; checked on a grid, ties skipped)\n")
    n = int(round(1 / step)) - 1
    grid = [(i * step, j * step) for i in range(1, n + 1) for j in range(1, n + 1)]
    fns = {"avg": poolib.avg, "geo": poolib.geo, "upco": poolib.upco, "mycin": poolib.mycin,
           "noisyor": poolib.noisy_or, "max": poolib.mx}
    mism = {k: 0 for k in fns}
    checked = 0
    for p1, p2 in grid:
        ref = p1 + p2 - 1
        if abs(ref) < 1e-9:
            continue
        checked += 1
        for k, f in fns.items():
            if ((f(p1, p2) - 0.5) > 0) != (ref > 0):
                mism[k] += 1
    print(f"  grid points checked: {checked}")
    for k in fns:
        tag = "same decision as p1+p2>1 everywhere" if mism[k] == 0 else "DIFFERENT decisions"
        print(f"    {k:8s} mismatches: {mism[k]:6d}   ({tag})")
    print("  -> the four opinion pools are indistinguishable by even-stakes guessing;")
    print("     noisy-or and max genuinely decide differently (they accumulate evidence).\n")


def expense_table(rounds, pools, ratios):
    """Average expense per round: act (pay C=ratio) iff q > ratio, else pay 1 if the event
    happens. rounds: (p_true, outcome, p1, p2); pools: name -> fn(p1,p2,p_true) -> q."""
    cost = {name: {r: 0.0 for r in ratios} for name in pools}
    for p_true, out, p1, p2 in rounds:
        for name, fn in pools.items():
            q = fn(p1, p2, p_true)
            for r in ratios:
                cost[name][r] += r if q > r else (1.0 if out else 0.0)
    n = len(rounds)
    return {name: {r: cost[name][r] / n for r in ratios} for name in pools}


def show(table, ratios):
    print("    pool     | " + "  ".join(f"C/L={r:.2f}" for r in ratios))
    oracle = table["oracle"]
    for name in sorted(table, key=lambda k: sum(table[k].values())):
        cells = []
        for r in ratios:
            c = table[name][r]
            mark = "*" if abs(c - min(table[k][r] for k in table)) < 5e-5 else " "
            cells.append(f"{c:.4f}{mark}")
        print(f"    {name:8s} | " + "  ".join(cells))
    print("    (* = best in column; oracle is the achievable floor)")


def main(T=200_000):
    part0_threshold_half_equivalence()

    ratios = [0.2, 0.35, 0.5, 0.65, 0.8]

    print("=== Part 1, World A: separate experts, prior 1/2 (true probability = upco) ===")
    print(f"average expense per animal, N={T} (lower is better)\n")
    show(expense_table(world_A(T=T, prior=0.5), std_pools(0.5), ratios), ratios)
    print("  -> upco sits on the oracle floor at every cost ratio. At C/L=0.5 the four pools")
    print("     tie exactly (part 0); at 0.2 and 0.8 the timid pools (avg, geo) pay extra.\n")

    print("=== Part 2, World B: mixture world (true probability = average) ===")
    print(f"average expense per animal, N={T} (lower is better)\n")
    show(expense_table(world_B(T=T), std_pools(0.5), ratios), ratios)
    print("  -> now avg sits on the floor and the overconfident pools (upco, noisy-or) pay")
    print("     extra at the outer ratios. Which pool 'wins' is set by the situation, at")
    print("     every stake level -- the correspondence map, priced in expected expense.")
    print("     (geo happens to tie avg here: on these five habitat pairs it never lands on")
    print("     a different side of the tested thresholds; the Kelly experiment separates")
    print("     them through bet size.)")


if __name__ == "__main__":
    main()
