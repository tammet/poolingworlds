#!/usr/bin/env python3
"""
Betting without knowing the world: which pooling rule to follow when the situation is unknown.

In decide_kelly_scenarios.py the bettor knows which world she is in and follows the pooling
rule built for it; each rule wins in its own world. Here she knows only that the world is one
of the N concrete worlds from that file, all equally likely, and she must pick one pooling
rule and stick to it. Two versions of "unknown":

  (a) series version: the world is drawn once at the start and stays fixed for the whole
      betting series. The bettor lives with one column of the growth matrix; which column,
      she learns only from her bank account.
  (b) round version: the world is drawn again before every observation. Each round is a
      fresh lottery over mechanisms.

For a fixed non-adaptive rule the expected growth is the same in (a) and (b): both equal the
plain average of the per-world growth rates. What differs is risk (in (a) the bettor is stuck
with the worst column if she is unlucky) and what the best possible rule is: in (b) the true
probability of the event given the two numbers is the average of the per-world truths, so the
average-of-world-formulas rule ("mixworld" below) is the best a world-ignorant bettor can do.
The gap between mixworld and the world-knowing oracle is the money value of knowing the world.

Which worlds can be mixed. The nine worlds use two different report interfaces. In worlds
1-5 and 8 both numbers say how strongly a source supports the event (probability scale); in
worlds 6, 7 and 9 the first number supports the event and the second speaks against it.
A single formula applied blindly across both interfaces would compare scale conventions, not
pooling quality, so we run two separate mixtures:

  group A: mixture(avg), experts(upco), overconfident(geo), separate triggers(noisy-or),
           redundant triggers(max), shared-luck proofs(cumul, lam=0.5) -- six worlds.
  group B: refutation(mycin), independent against-evidence(problog), contradicting
           witnesses(DS pignistic) -- three worlds.

One change against decide_kelly_scenarios.py: there the worlds draw their two numbers from
different distributions (worlds 1 and 8 from short fixed lists of pairs), so a bettor could
recognize the world from the numbers alone and "unknown" would be fiction. Here every group-A
world draws the two numbers the same way (independent uniforms on 0.15..0.85) and only the
outcome mechanism differs; group B already has one common input law. For the mixture and
trigger worlds the original mechanisms run unchanged; for the expert worlds (upco, geo) the
outcome is drawn from the mechanism's conditional law given the reports (a Bernoulli with the
world's truth formula), which is the same bet-relevant object.

Reported: the per-world growth matrix (columns = fixed worlds of situation (a)), its mean
(= expected growth when the fixed world is drawn at random) and worst column, and the mixed
per-round run of situation (b) with its ranking. The script checks that the matrix mean and
the mixed-run growth agree and that mixworld wins the mixed run.
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib
from decide_kelly_betting import growth_table, clip
from corr_mycin_probe import poisson

T_WORLD = 100_000     # rounds per fixed-world series (situation a)
T_MIXED = 300_000     # rounds of the mixed run (situation b)


# ------------------------------------------------------------------ group A: six worlds
# Each world's truth given the two numbers is one pooling formula of them.
TRUTHS_A = {
    "avg":     poolib.avg,
    "upco":    poolib.upco,
    "geo":     poolib.geo,
    "noisyor": poolib.noisy_or,
    "max":     poolib.mx,
    "cumul":   lambda p, q: poolib.cumul(p, q, 0.5),
}

def outcome_A(world, p1, p2):
    """Run the world's mechanism once for reports (p1, p2)."""
    if world == "avg":                      # one of two habitats is real
        rate = p1 if random.random() < 0.5 else p2
        return random.random() < rate
    if world == "noisyor":                  # two separate triggers
        return (random.random() < p1) or (random.random() < p2)
    if world == "max":                      # redundant triggers, one shared draw
        return random.random() < max(p1, p2)
    if world == "cumul":                    # shared luck with probability lam = 0.5
        if random.random() < 0.5:
            u = random.random()
            return u < p1 or u < p2
        return (random.random() < p1) or (random.random() < p2)
    # upco, geo: expert worlds -- outcome from the conditional law given the reports
    return random.random() < TRUTHS_A[world](p1, p2)

def rounds_A(world, T, seed):
    random.seed(seed)
    rounds = []
    for _ in range(T):
        p1, p2 = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        out = outcome_A(world, p1, p2)
        rounds.append((TRUTHS_A[world](p1, p2), out, p1, p2))
    return rounds

def rounds_A_mixed(T, seed):
    random.seed(seed)
    names = WORLDS_A
    rounds = []
    for _ in range(T):
        w = random.choice(names)
        p1, p2 = random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)
        out = outcome_A(w, p1, p2)
        rounds.append((TRUTHS_A[w](p1, p2), out, p1, p2))
    return rounds

def mixworld_A(p, q):
    return sum(f(p, q) for f in TRUTHS_A.values()) / len(TRUTHS_A)

POOLS_A = {
    "oracle":   lambda p, q, pt: pt,
    "mixworld": lambda p, q, pt: mixworld_A(p, q),
    "avg":      lambda p, q, pt: poolib.avg(p, q),
    "geo":      lambda p, q, pt: poolib.geo(p, q),
    "cumul":    lambda p, q, pt: poolib.cumul(p, q, 0.5),
    "max":      lambda p, q, pt: poolib.mx(p, q),
    "noisyor":  lambda p, q, pt: poolib.noisy_or(p, q),
    "upco":     lambda p, q, pt: poolib.upco(p, q),
    "mycin":    lambda p, q, pt: poolib.mycin(p, q),
    "base":     lambda p, q, pt: 0.5,
}


# ------------------------------------------------------------------ group B: three worlds
# One for-number a and one against-number b, drawn the same way in all three worlds
# (a uniform on 0.5..0.95, b uniform on 0.05..a-0.05), mechanisms as in
# decide_kelly_scenarios.py.
def outcome_B(world, a, b):
    if world == "mycin":                     # each against-sign refutes one for-sign
        la, lb = -math.log(1 - a), -math.log(1 - b)
        n_for = poisson(la)
        return sum(random.random() >= lb / la for _ in range(n_for)) >= 1
    if world == "problog":                  # against-evidence independent of for-evidence
        return (random.random() < a) and not (random.random() < b)
    # DS witnesses: both-reliable impossible (redraw); neither-reliable -> fair coin
    while True:
        r1, r2 = random.random() < a, random.random() < b
        if not (r1 and r2):
            break
    return r1 or ((not r2) and random.random() < 0.5)

TRUTHS_B = {
    "mycin":   lambda a, b: (a - b) / (1 - b),
    "problog": lambda a, b: a * (1 - b),
    "ds":      poolib.ds_pignistic,
}

# fixed world order (pins the seed <-> world pairing and the mixed-round sequence,
# so results stay reproducible independently of the rule names)
WORLDS_A = ["avg", "cumul", "geo", "max", "noisyor", "upco"]
WORLDS_B = ["mycin", "ds", "problog"]

def rounds_B(world, T, seed):
    random.seed(seed)
    rounds = []
    for _ in range(T):
        a = random.uniform(0.5, 0.95)
        b = random.uniform(0.05, a - 0.05)
        out = outcome_B(world, a, b)
        rounds.append((TRUTHS_B[world](a, b), out, a, b))
    return rounds

def rounds_B_mixed(T, seed):
    random.seed(seed)
    names = WORLDS_B
    rounds = []
    for _ in range(T):
        w = random.choice(names)
        a = random.uniform(0.5, 0.95)
        b = random.uniform(0.05, a - 0.05)
        out = outcome_B(w, a, b)
        rounds.append((TRUTHS_B[w](a, b), out, a, b))
    return rounds

def mixworld_B(a, b):
    return sum(f(a, b) for f in TRUTHS_B.values()) / len(TRUTHS_B)

POOLS_B = {
    "oracle":   lambda a, b, pt: pt,
    "mixworld": lambda a, b, pt: mixworld_B(a, b),
    "mycin":    lambda a, b, pt: (a - b) / (1 - b),
    "problog":  lambda a, b, pt: a * (1 - b),
    "pignist":  lambda a, b, pt: poolib.ds_pignistic(a, b),
    "ds_bel":   lambda a, b, pt: poolib.ds_bel(a, b),
    "ds_pl":    lambda a, b, pt: poolib.ds_pl(a, b),
    "naive":    lambda a, b, pt: max(a - b, 0.0),
    "base":     lambda a, b, pt: 0.5,
}


# ------------------------------------------------------------------ reporting
def matrix(world_names, rounds_fn, pools, seed0):
    """Situation (a): one long fixed-world series per world. Returns
    {pool: {world: growth}}."""
    per = {name: {} for name in pools}
    for i, w in enumerate(world_names):
        rates = growth_table(rounds_fn(w, T_WORLD, seed0 + i), pools, prior=0.5)
        for name in pools:
            per[name][w] = rates[name]
    return per

def show_matrix(per, world_names):
    head = "  ".join(f"{w:>8s}" for w in world_names)
    print(f"    {'pool':9s} {head}      mean     worst")
    rows = []
    for name, cols in per.items():
        mean = sum(cols.values()) / len(cols)
        worst = min(cols.values())
        rows.append((name, cols, mean, worst))
    for name, cols, mean, worst in sorted(rows, key=lambda r: -r[2]):
        cells = "  ".join(f"{cols[w]:+8.4f}" for w in world_names)
        print(f"    {name:9s} {cells}  {mean:+8.4f}  {worst:+8.4f}")
    return {name: mean for name, _, mean, _ in rows}

def show_mixed(rates):
    for name in sorted(rates, key=lambda k: -rates[k]):
        print(f"    {name:9s} growth {rates[name]:+.4f} bits/round"
              f"   given up vs oracle {rates['oracle'] - rates[name]:.4f}")

def report(label, world_names, rounds_fn, rounds_mixed_fn, pools, seed0):
    print(f"--- {label} ---\n")
    print("  situation (a): world drawn once, fixed for the series")
    print("  (growth in bits/round for each fixed world; ordered by mean = expected growth")
    print("   over the world draw; worst = the column an unlucky bettor is stuck with)\n")
    per = matrix(world_names, rounds_fn, pools, seed0)
    means = show_matrix(per, world_names)

    print("\n  situation (b): world drawn again before every round\n")
    rates = growth_table(rounds_mixed_fn(T_MIXED, seed0 + 99), pools, prior=0.5)
    show_mixed(rates)

    ranked = sorted((n for n in rates if n != "oracle"), key=lambda n: -rates[n])
    winner = ranked[0]
    ok = "as predicted" if winner == "mixworld" else "UNEXPECTED (predicted mixworld)"
    print(f"\n  -> best world-ignorant rule in (b): {winner} -- {ok}")
    drift = max(abs(means[n] - rates[n]) for n in rates)
    print(f"  -> matrix mean vs mixed run: largest difference {drift:.4f} bits/round"
          f" (theory: equal for a fixed rule; difference is sampling noise)")
    print(f"  -> money value of knowing the world: oracle - mixworld ="
          f" {rates['oracle'] - rates['mixworld']:.4f} bits/round\n")
    return rates, means


def main():
    print("=== betting when the world is unknown (equal chances over the known worlds) ===")
    print("(even odds; mixworld = average of the worlds' truth formulas; base = q of 1/2)\n")
    report("group A: both numbers support the event (six worlds)",
           WORLDS_A, rounds_A, rounds_A_mixed, POOLS_A, seed0=11)
    report("group B: for-number and against-number (three worlds)",
           WORLDS_B, rounds_B, rounds_B_mixed, POOLS_B, seed0=31)


if __name__ == "__main__":
    main()
