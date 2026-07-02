#!/usr/bin/env python3
"""
Using the pooled number to WIN MONEY: a bettor who pools two reports and bets on the outcome.

Story. A bookmaker offers bets on whether the next animal is a bird, at odds that are fair for
someone who knows only the base rate. A bettor gets the two observers' probability reports,
pools them into one number q, and each round splits her stake in proportion q on "bird" and
1-q on "not bird" (the classic Kelly / horse-race scheme: the winning side pays out at the
bookmaker's odds). Her long-run wealth growth measures how good her pooling rule is: betting
with the true probability grows wealth fastest, and any other number costs a growth rate equal
to its KL divergence from the truth (Kelly 1956; Cover & Thomas, Elements of Information
Theory, ch. 6). "Money won" is the pragmatic face of the log score.

Why betting separates pools where counting correct guesses does not: a guess only uses which
side of 1/2 the number falls on, and (decide_cost_loss.py, part 0) avg, geometric, upco and
mycin always fall on the SAME side of 1/2. Bet SIZING uses the whole number, so an
overconfident or underconfident pool loses money even when it picks the same side.

Three worlds, same bettor, different true situations:
  A  two experts examine the animal separately; birds and non-birds equally common.
     True probability = upco of the reports -> the upco bettor matches the oracle; avg and
     geo bet too timidly and grow slower.
  A' same experts, but birds are rare (prior 0.3) and the odds are priced for that.
     Plain upco double-counts the prior and over-bets; the prior-corrected odds pool matches
     the oracle.
  B  mixture world: the site is one of two habitats (only one is real) with known bird-rates
     p1, p2. True probability = average -> the avg bettor matches the oracle; the upco bettor
     is overconfident and bleeds money.

Reported: per-round growth in bits (average log2 of the wealth multiplier), the growth given
up relative to the oracle (this gap IS the average KL divergence of the pool from the truth --
the money price of using the wrong pool), and the wealth multiple after 100 typical rounds.
Growth 0 = treading water (the bettor who just bets the base rate); negative = losing money in
a game where the informed bettor profits.
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib

EPS = 1e-6

def clip(q):
    return min(max(q, EPS), 1 - EPS)


def growth_table(rounds, pools, prior):
    """rounds: list of (p_true, outcome, p1, p2).
    pools: name -> fn(p1, p2, p_true) -> q.
    The bettor splits her stake q : (1-q); the bookmaker pays 1/prior per unit on 'bird' and
    1/(1-prior) on 'not bird' (fair odds for the base rate).
    Per-round log2 growth = log2(q_outcome / prior_outcome)."""
    g = {name: 0.0 for name in pools}
    for p_true, out, p1, p2 in rounds:
        for name, fn in pools.items():
            q = clip(fn(p1, p2, p_true))
            q_out = q if out else 1 - q
            pi_out = prior if out else 1 - prior
            g[name] += math.log2(q_out / pi_out)
    n = len(rounds)
    return {name: g[name] / n for name in pools}


def show(rates):
    oracle = rates.get("oracle", max(rates.values()))
    for name in sorted(rates, key=lambda k: -rates[k]):
        r = rates[name]
        wealth = 2 ** (100 * r)
        w = f"{wealth:,.2f}x" if 0.01 <= wealth <= 1e6 else f"{wealth:.2e}x"
        print(f"    {name:9s} growth {r:+.4f} bits/round   given up vs oracle {oracle - r:.4f}"
              f"   wealth after 100 rounds ~ {w}")


def std_pools(prior):
    """The candidate pooling rules, plus the oracle (true probability) and the base-rate bettor."""
    return {
        "oracle":  lambda p1, p2, pt: pt,
        "upco":    lambda p1, p2, pt: poolib.upco(p1, p2),
        "avg":     lambda p1, p2, pt: poolib.avg(p1, p2),
        "geo":     lambda p1, p2, pt: poolib.geo(p1, p2),
        "mycin":   lambda p1, p2, pt: poolib.mycin(p1, p2),
        "noisyor": lambda p1, p2, pt: poolib.noisy_or(p1, p2),
        "base":    lambda p1, p2, pt: prior,
    }


def world_A(T=200_000, seed=1, prior=0.5):
    """Two separate experts with per-round accuracies drawn U(0.55,0.95), calibrated reports.
    Exact posterior = prior-corrected odds pool of the reports (= upco when prior = 1/2)."""
    random.seed(seed)
    rounds = []
    for _ in range(T):
        H = random.random() < prior
        rep = []
        for _ in range(2):
            acc = random.uniform(0.55, 0.95)
            sig = H if random.random() < acc else not H
            lh, ln = (acc, 1 - acc) if sig else (1 - acc, acc)   # P(sig|bird), P(sig|not)
            rep.append(lh * prior / (lh * prior + ln * (1 - prior)))
        p1, p2 = rep
        p_true = poolib.bayes_odds_pool([p1, p2], prior)
        rounds.append((p_true, H, p1, p2))
    return rounds


def world_B(T=200_000, seed=1):
    """Mixture world: per round pick a known pair of habitat bird-rates, flip which habitat is
    real (1/2 each), then draw the animal. True P(bird) = (p1+p2)/2."""
    random.seed(seed)
    pairs = [(0.30, 0.80), (0.15, 0.65), (0.60, 0.90), (0.20, 0.50), (0.40, 0.70)]
    rounds = []
    for _ in range(T):
        p1, p2 = random.choice(pairs)
        rate = p1 if random.random() < 0.5 else p2
        H = random.random() < rate
        rounds.append(((p1 + p2) / 2, H, p1, p2))
    return rounds


def main():
    print("=== Kelly betting: wealth growth of a bettor following each pooling rule ===")
    print("(odds fair for the base rate; bettor splits her stake by the pooled q;")
    print(" oracle = betting the true per-round probability; base = betting the base rate)\n")

    print("World A: two separate experts, birds and non-birds equally common (prior 1/2)")
    show(growth_table(world_A(prior=0.5), std_pools(0.5), prior=0.5))
    print("  -> upco = oracle: the two reports multiply. avg and geo pick the same side but")
    print("     bet too timidly; noisy-or over-bets. Money shows what guess-counting cannot.\n")

    print("World A': same experts, but birds are rare (prior 0.3; odds priced for 0.3)")
    pools = std_pools(0.3)
    pools["oddspool"] = lambda p1, p2, pt: poolib.bayes_odds_pool([p1, p2], 0.3)
    show(growth_table(world_A(prior=0.3), pools, prior=0.3))
    print("  -> the prior-corrected odds pool = oracle; plain upco double-counts the rarity")
    print("     of birds, over-bets, and loses growth.\n")

    print("World B: mixture world (two habitats, only one real; true P = average)")
    show(growth_table(world_B(), std_pools(0.5), prior=0.5))
    print("  -> avg = oracle here; upco/mycin/noisy-or are overconfident about the mixture and")
    print("     grow slower or shrink. The best rule to bet with is the one matching the")
    print("     situation -- the corr_* correspondence map, now measured in money.")


if __name__ == "__main__":
    main()
