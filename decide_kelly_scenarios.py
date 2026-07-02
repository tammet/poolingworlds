#!/usr/bin/env python3
"""
Every pool has its world: for each pooling function in this suite, a betting scenario where
following THAT pool wins the most money.

Setup as in decide_kelly_betting.py: each round two numbers arrive, the bettor pools them into
q and splits her stake q : (1-q) at the bookmaker's fixed odds. Betting with the true
conditional probability of the world grows wealth fastest, and any other number loses growth
at rate KL(truth || number). So "which pool is monetarily best" = "which pool equals the true
probability of this world" -- and each world below is built so that a different pool is that
truth. The ranking does not depend on the bookmaker's odds (growth differences cancel them),
so all worlds use even odds; the base bettor (q = 1/2) always has growth exactly 0.

The eight worlds and their winners:

  1 avg      mixture: the site is one of two habitats, rates p1, p2, only one real.
  2 upco     two experts examine the animal separately, calibrated reports, even prior.
  3 geo      two overconfident experts: each reports DOUBLED log-odds (a well-documented
             bias -- overprecision; cf. Ranjan & Gneiting 2010 on recalibrating pools).
             Geometric pooling averages log-odds, exactly undoing the doubling.
  4 noisyor  two separate triggers with rates p1, p2; the event is "at least one fires".
  5 max      two redundant triggers (the stronger one's fires include the weaker one's).
  6 mycin    refutation world: for-signs arrive (chance a that at least one appears);
             every against-sign REFUTES one specific for-sign (against-observer's marginal
             chance of finding one is b); the event is "at least one for-sign survives".
             True P = (a-b)/(1-b) -- the mycin conflict rule, exactly. This is the coupling
             the corr_mycin_probe was missing: independence between the for- and
             against-processes gives problog; complete one-on-one refutation gives mycin --
             the same split as independence -> noisy-or vs redundancy -> max in world 4/5.
  7 problog  same reports, but the against-evidence is INDEPENDENT of the for-evidence and
             the event is "a for-sign appears and no against-sign does": true P = a(1-b).
  8 cumul    two proofs of one fact that share their luck with a known probability lam
             (the common-shock model): true P = cumul(c1, c2, a=1-lam).
  9 DS       two contradicting witnesses of known reliabilities m1, m2, who cannot both be
             reliable (both-reliable rounds are impossible and redrawn = Dempster's
             renormalization); undecided rounds resolved by a fair coin. True P = the
             pignistic probability BetP -- Smets' "bet with BetP" rule. bel and pl are the
             lower/upper bounds and mis-bet; see corr_dempster_shafer.py.

Worlds 6 and 7 compare the three conflict rules on the same inputs (naive difference a-b,
problog a(1-b), mycin (a-b)/(1-b)); the naive CONFER-2021 rule wins in neither world.

A plain-language, detailed description of the betting mechanism and of each world is in
../MEMO_betting_worlds_plain.md.
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib
from decide_kelly_betting import growth_table, show, std_pools, world_A, world_B
from corr_mycin_probe import poisson

T = 100_000


def logit(p):  return math.log(p / (1 - p))
def sigmoid(x): return 1 / (1 + math.exp(-x))


def world_geo(T=T, seed=2):
    """Two conditionally independent signals; each expert reports with doubled log-odds."""
    random.seed(seed)
    rounds = []
    for _ in range(T):
        H = random.random() < 0.5
        reps, l_sum = [], 0.0
        for _ in range(2):
            acc = random.uniform(0.6, 0.9)
            sig = H if random.random() < acc else not H
            l = logit(acc) if sig else -logit(acc)     # this signal's true log-odds
            l_sum += l
            reps.append(sigmoid(2 * l))                # overconfident: doubled log-odds
        rounds.append((sigmoid(l_sum), H, reps[0], reps[1]))
    return rounds


def world_or(T=T, seed=2, coupled=False):
    """Two triggers with rates p1,p2; event = at least one fires.
    coupled=False: independent -> truth = noisy-or; True: redundant -> truth = max."""
    random.seed(seed)
    rounds = []
    for _ in range(T):
        p1, p2 = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        if coupled:
            u = random.random()
            out = u < max(p1, p2)
            rounds.append((max(p1, p2), out, p1, p2))
        else:
            out = (random.random() < p1) or (random.random() < p2)
            rounds.append((poolib.por(p1, p2), out, p1, p2))
    return rounds


def world_conflict(T=T, seed=2, refutation=True):
    """For-signs ~ Poisson(la), a = P(at least one). refutation=True: each against-sign
    refutes one for-sign (thinning; against marginal = b); event = a for-sign survives;
    truth = (a-b)/(1-b) = mycin. refutation=False: against-evidence independent; event =
    for-sign appears and no against-sign; truth = a(1-b) = problog."""
    random.seed(seed)
    rounds = []
    for _ in range(T):
        a = random.uniform(0.5, 0.95)
        b = random.uniform(0.05, a - 0.05)
        if refutation:
            la, lb = -math.log(1 - a), -math.log(1 - b)
            n_for = poisson(la)
            survivors = sum(random.random() >= lb / la for _ in range(n_for))
            out = survivors >= 1
            rounds.append(((a - b) / (1 - b), out, a, b))
        else:
            out = (random.random() < a) and not (random.random() < b)
            rounds.append((a * (1 - b), out, a, b))
    return rounds


def world_ds(T=T, seed=2):
    """Contradicting witnesses: witness 1 (for) reliable w.p. m1, witness 2 (against) w.p. m2,
    independently; both-reliable is impossible (redraw); neither-reliable -> fair coin.
    Truth per round = ds_pignistic(m1, m2)."""
    random.seed(seed)
    rounds = []
    for _ in range(T):
        m1 = random.uniform(0.5, 0.95)
        m2 = random.uniform(0.05, m1 - 0.05)          # keep m2 < m1, as in worlds 6/7
        while True:
            r1, r2 = random.random() < m1, random.random() < m2
            if not (r1 and r2):
                break
        out = r1 or ((not r2) and random.random() < 0.5)
        rounds.append((poolib.ds_pignistic(m1, m2), out, m1, m2))
    return rounds


def world_cumul(T=T, seed=2, lam=0.5):
    """Common-shock model with known sharing probability lam: with prob lam the two proofs
    share one draw, else independent; truth = cumul(c1, c2, 1-lam)."""
    random.seed(seed)
    pairs = [(0.5, 0.8), (0.3, 0.7), (0.6, 0.6), (0.4, 0.9)]
    rounds = []
    for _ in range(T):
        c1, c2 = random.choice(pairs)
        if random.random() < lam:
            u = random.random(); out = u < c1 or u < c2
        else:
            out = (random.random() < c1) or (random.random() < c2)
        rounds.append((poolib.cumul(c1, c2, 1 - lam), out, c1, c2))
    return rounds


CONFLICT_POOLS = {
    "oracle":  lambda a, b, pt: pt,
    "mycin":   lambda a, b, pt: (a - b) / (1 - b),        # mycin conflict rule on strengths
    "problog": lambda a, b, pt: a * (1 - b),
    "naive":   lambda a, b, pt: max(a - b, 0.0),          # CONFER-2021 naive difference
    "base":    lambda a, b, pt: 0.5,
}

DS_POOLS = {
    "oracle":   lambda a, b, pt: pt,
    "pignist":  lambda a, b, pt: poolib.ds_pignistic(a, b),
    "ds_bel":   lambda a, b, pt: poolib.ds_bel(a, b),
    "ds_pl":    lambda a, b, pt: poolib.ds_pl(a, b),
    "mycin":    lambda a, b, pt: (a - b) / (1 - b),
    "problog":  lambda a, b, pt: a * (1 - b),
    "naive":    lambda a, b, pt: max(a - b, 0.0),
    "base":     lambda a, b, pt: 0.5,
}

CUMUL_POOLS = {
    "oracle":  lambda c1, c2, pt: pt,
    "cumul":   lambda c1, c2, pt: poolib.cumul(c1, c2, 0.5),   # knows lam = 0.5
    "max":     lambda c1, c2, pt: poolib.mx(c1, c2),
    "noisyor": lambda c1, c2, pt: poolib.noisy_or(c1, c2),
    "avg":     lambda c1, c2, pt: poolib.avg(c1, c2),
    "base":    lambda c1, c2, pt: 0.5,
}


def run(title, rounds, pools, expect):
    print(title)
    rates = growth_table(rounds, pools, prior=0.5)
    show(rates)
    ranked = sorted((n for n in rates if n != "oracle"), key=lambda n: -rates[n])
    winner = ranked[0]
    margin = rates[winner] - rates[ranked[1]]
    ok = "as predicted" if winner == expect else f"UNEXPECTED (predicted {expect})"
    print(f"  -> best pool: {winner} (+{margin:.4f} bits/round over runner-up) -- {ok}\n")
    return winner, expect


def main():
    print("=== every pool has its world: the monetarily best pool per betting scenario ===")
    print("(even odds throughout; ranking is independent of the odds; base = q of 1/2)\n")
    results = []
    results.append(run("World 1 -- mixture (two habitats, one real): truth = avg",
                       world_B(T=T), std_pools(0.5), "avg"))
    results.append(run("World 2 -- separate calibrated experts, even prior: truth = upco",
                       world_A(T=T, prior=0.5), std_pools(0.5), "upco"))
    results.append(run("World 3 -- overconfident experts (doubled log-odds): truth = geo",
                       world_geo(), std_pools(0.5), "geo"))
    results.append(run("World 4 -- two separate triggers, bet on 'at least one': truth = noisy-or",
                       world_or(coupled=False), std_pools(0.5), "noisyor"))
    results.append(run("World 5 -- two redundant triggers, bet on 'at least one': truth = max",
                       world_or(coupled=True),
                       {**std_pools(0.5), "max": lambda p, q, pt: poolib.mx(p, q)}, "max"))
    results.append(run("World 6 -- refutation (each against-sign kills one for-sign): truth = mycin",
                       world_conflict(refutation=True), CONFLICT_POOLS, "mycin"))
    results.append(run("World 7 -- independent against-evidence: truth = problog",
                       world_conflict(refutation=False), CONFLICT_POOLS, "problog"))
    results.append(run("World 8 -- proofs sharing luck with known probability: truth = cumul",
                       world_cumul(), CUMUL_POOLS, "cumul"))
    results.append(run("World 9 -- contradicting witnesses (Dempster-Shafer): truth = pignistic",
                       world_ds(), DS_POOLS, "pignist"))

    print("=== summary ===")
    allok = all(w == e for w, e in results)
    for (w, e), n in zip(results, range(1, len(results) + 1)):
        print(f"  world {n}: winner {w}" + ("" if w == e else f"  (predicted {e})"))
    print("\n-> " + ("each pooling function is the best money-maker in its own world." if allok
                     else "some worlds did not match the prediction -- inspect above."))
    print("   The map generative situation -> optimal pool (the corr_* experiments) is the")
    print("   same map as betting scenario -> the pool to follow for maximal wealth.")


if __name__ == "__main__":
    main()
