#!/usr/bin/env python3
"""
The log-odds family with a knob: extremized (symmetric alpha) and weighted (per-expert)
pools, and the situations where each knob setting wins the betting game.

The family. Convert each report to log-odds, take a weighted sum, convert back:
    pooled log-odds = w1*logit(p1) + w2*logit(p2).
Members already in the suite: geo is (0.5, 0.5), upco is (1, 1). The symmetric case
w1 = w2 = alpha is the "extremized" pool of the forecasting-tournament literature
(alpha > 1 pushes the result away from 1/2; Baron et al. 2014, Satopaa et al. 2014).

When is which setting right? Two separate forces move the best alpha away from 1:
  * report distortion -- an expert who exaggerates multiplies her honest log-odds by some
    k > 1; a timid expert has k < 1. The fix is alpha = 1/k: halve doubled claims (world 3
    of decide_kelly_scenarios, alpha = 1/2), or double timid claims (extremizing, alpha = 2).
    These are exact: the alpha-bettor matches the oracle.
  * shared evidence -- if the experts partly saw the same thing, their log-odds sum counts
    the shared part twice, pushing the best alpha below 1. This fix is NOT exact: no single
    alpha reproduces the truth, because the right correction depends on how the shared and
    private parts point, not just on their sizes. The sweep below shows the best alpha in
    between, with a remaining gap to the oracle -- the honest price of a one-knob rule.
Asymmetric distortion (one expert exaggerates, the other is timid) needs per-expert weights
w_i = 1/k_i; no symmetric alpha and none of the standard pools match.

The same weighted idea on the linear side: in a mixture world (decide_kelly_scenarios
world 1) with UNEQUAL situation probabilities, the right rule is the weighted average with
weights = the situation probabilities -- the classical interpretation of linear-pool weights.

In all parts the bettor is TOLD the distortion factors / overlap / mixture weights -- the
knob is knowledge about the situation, like cumulation's sharing probability. Estimating the
knob from past data (Cooke-style performance weighting) is a separate, future experiment.
Betting mechanism and reading of the numbers: ../MEMO_betting_worlds_plain.md.
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib
from decide_kelly_betting import growth_table, show

T = 100_000

def logit(p):   return math.log(p / (1 - p))
def sigmoid(x): return 1 / (1 + math.exp(-x))


# ---------------------------------------------------------------- round generators
def rounds_private(T=T, seed=3, k=1.0):
    """Two conditionally independent signals; each expert reports log-odds distorted by k
    (k=1 calibrated, k>1 exaggerating, k<1 timid). Truth = sum of honest log-odds."""
    random.seed(seed)
    out = []
    for _ in range(T):
        H = random.random() < 0.5
        ls, reps = 0.0, []
        for _ in range(2):
            acc = random.uniform(0.6, 0.9)
            sig = H if random.random() < acc else not H
            l = logit(acc) if sig else -logit(acc)
            ls += l
            reps.append(sigmoid(k * l))
        out.append((sigmoid(ls), H, reps[0], reps[1]))
    return out


def rounds_shared(T=T, seed=3):
    """Both experts report the same single signal (fully shared evidence)."""
    random.seed(seed)
    out = []
    for _ in range(T):
        H = random.random() < 0.5
        acc = random.uniform(0.6, 0.9)
        sig = H if random.random() < acc else not H
        l = logit(acc) if sig else -logit(acc)
        p = sigmoid(l)
        out.append((p, H, p, p))
    return out


def rounds_halfshared(T=T, seed=3, acc_s=0.8, acc_p=0.75):
    """Each expert sees one shared signal (accuracy acc_s) and one private signal (acc_p),
    and reports the calibrated posterior of what she saw. Fixed accuracies, so the truth
    P(H | both signals) is exact and reachable from the reports -- just not by any single
    alpha."""
    random.seed(seed)
    L_s, L_p = logit(acc_s), logit(acc_p)
    out = []
    for _ in range(T):
        H = random.random() < 0.5
        # each signal points toward the truth with its accuracy (sign +1) or away (-1)
        sgn_s  = 1 if random.random() < acc_s else -1
        sgn_p1 = 1 if random.random() < acc_p else -1
        sgn_p2 = 1 if random.random() < acc_p else -1
        d = 1 if H else -1
        ls, lp1, lp2 = d * sgn_s * L_s, d * sgn_p1 * L_p, d * sgn_p2 * L_p
        r1, r2 = sigmoid(ls + lp1), sigmoid(ls + lp2)
        out.append((sigmoid(ls + lp1 + lp2), H, r1, r2))
    return out


def rounds_mixture(T=T, seed=3, w=0.7):
    """Mixture world with unequal situation probabilities: habitat 1 with probability w."""
    random.seed(seed)
    pairs = [(0.30, 0.80), (0.15, 0.65), (0.60, 0.90), (0.20, 0.50)]
    out = []
    for _ in range(T):
        p1, p2 = random.choice(pairs)
        rate = p1 if random.random() < w else p2
        ev = random.random() < rate
        out.append((w * p1 + (1 - w) * p2, ev, p1, p2))
    return out


# ---------------------------------------------------------------- sweep helper
ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5]

def alpha_sweep(title, rounds, predicted):
    print(title)
    pools = {"oracle": lambda p, q, pt: pt}
    for a in ALPHAS:
        pools[f"a={a:g}"] = (lambda a: lambda p, q, pt: poolib.alpha_logodds(p, q, a))(a)
    rates = growth_table(rounds, pools, prior=0.5)
    oracle = rates["oracle"]
    best = max((n for n in rates if n != "oracle"), key=lambda n: rates[n])
    print("    alpha : " + "  ".join(f"{a:g}" for a in ALPHAS))
    print("    growth: " + "  ".join(f"{rates[f'a={a:g}']:.3f}" for a in ALPHAS))
    gap = oracle - rates[best]
    print(f"    best alpha = {best[2:]} (predicted {predicted}); gap to oracle {gap:.4f} bits/round"
          + ("  -- exact" if gap < 2e-3 else "  -- NOT exact: no alpha reaches the oracle"))
    print()


def main():
    print("=== the log-odds family with a knob: when each alpha / weight wins ===")
    print("(same betting game as decide_kelly_scenarios; growth in bits/round)\n")

    print("Part 1: symmetric alpha, four regimes")
    alpha_sweep("  1a. private evidence, calibrated reports -> alpha 1 (= upco) exact",
                rounds_private(k=1.0), "1")
    alpha_sweep("  1b. fully shared evidence -> alpha 0.5 (= geo) exact",
                rounds_shared(), "0.5")
    alpha_sweep("  1c. private evidence, timid reports (k = 0.5) -> alpha 2 (extremizing) exact",
                rounds_private(k=0.5), "2")
    alpha_sweep("  1d. half-shared evidence, calibrated -> best alpha strictly between, not exact",
                rounds_halfshared(), "between 0.5 and 1")

    print("Part 2: asymmetric distortion -> per-expert weights w_i = 1/k_i")
    print("  expert 1 exaggerates (k1 = 2), expert 2 is timid (k2 = 0.5); private evidence.")
    rounds = []
    random.seed(4)
    for _ in range(T):
        H = random.random() < 0.5
        ls, reps = 0.0, []
        for k in (2.0, 0.5):
            acc = random.uniform(0.6, 0.9)
            sig = H if random.random() < acc else not H
            l = logit(acc) if sig else -logit(acc)
            ls += l
            reps.append(sigmoid(k * l))
        rounds.append((sigmoid(ls), H, reps[0], reps[1]))
    pools = {
        "oracle":   lambda p, q, pt: pt,
        "weighted": lambda p, q, pt: poolib.wlogodds(p, q, 0.5, 2.0),   # 1/k1, 1/k2
        "upco":     lambda p, q, pt: poolib.upco(p, q),
        "geo":      lambda p, q, pt: poolib.geo(p, q),
        "avg":      lambda p, q, pt: poolib.avg(p, q),
        "base":     lambda p, q, pt: 0.5,
    }
    for a in (0.75, 1.0):
        pools[f"a={a:g}"] = (lambda a: lambda p, q, pt: poolib.alpha_logodds(p, q, a))(a)
    show(growth_table(rounds, pools, prior=0.5))
    print("  -> the (1/k1, 1/k2)-weighted pool matches the oracle; no symmetric alpha and")
    print("     none of the standard pools do.\n")

    print("Part 3: weighted linear pool in an unequal mixture (habitat 1 with probability 0.7)")
    pools = {
        "oracle": lambda p, q, pt: pt,
        "wavg.7": lambda p, q, pt: poolib.wavg(p, q, 0.7),
        "avg":    lambda p, q, pt: poolib.avg(p, q),
        "geo":    lambda p, q, pt: poolib.geo(p, q),
        "upco":   lambda p, q, pt: poolib.upco(p, q),
        "base":   lambda p, q, pt: 0.5,
    }
    show(growth_table(rounds_mixture(w=0.7), pools, prior=0.5))
    print("  -> weights in the linear pool = the situation probabilities; the equal-weight")
    print("     average pays for assuming the habitats are equally likely.")


if __name__ == "__main__":
    main()
