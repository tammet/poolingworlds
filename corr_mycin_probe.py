#!/usr/bin/env python3
"""
The MYCIN certainty-factor combination -- does it match a Monte-Carlo count?

The rule tested here is the REVISED MYCIN combination (van Melle's EMYCIN version, documented
in Buchanan & Shortliffe, Rule-Based Expert Systems, 1984): on strengths, agreement combines
as a + b - ab and conflict as (a - b)/(1 - b) for a > b. The original 1975 rule added
opposite-sign factors without the renormalizing denominator. See poolib.mycin.

Story. One observer collects signs that the animal is a bird; another collects signs that it is
not. MYCIN treats the two as opposing piles of signs that cancel one against the other. Whatever
is left over after cancelling decides the verdict, scaled by how much evidence is left. When both
observers collect signs for the same side, the piles add up instead, the same as two separate
observers (noisy-or).

Two parts:

  Agreement (both "for", strengths a, b):  mycin(a,b) = a + b - a*b = noisy-or.
     This matches a count: at least one of two separate "for" signs turns up (same as problog).

  Conflict ("for" a vs "against" b, a > b):  mycin_diff(a,b) = (a-b)/(1-b).
     Picture each observer as collecting signs over a fixed watch period: signs arrive now and
     then at some steady rate, and strength x is the chance that at least one sign turns up
     during the period. That rate is lambda = -ln(1-x) (so x = 1 - e^{-lambda}); a stronger
     observer has a higher arrival rate. In this picture the MYCIN conflict rule is
        (a-b)/(1-b) = 1 - e^{-(lambda_a - lambda_b)},
     that is: subtract the against-rate from the for-rate, sign for sign, and let the leftover
     net rate produce the result.

     Which observation counts give this number? Under INDEPENDENT for- and against-processes,
     none of the natural counts do (parts C-a, C-b2 below). But there IS a coupling that gives
     it exactly (part D): let every against-sign REFUTE one specific for-sign (the
     against-process is a thinned subset of the for-process), and count the rounds where
     at least one for-sign survives. Survivors form a Poisson process of rate
     lambda_a - lambda_b, so P(at least one survives) = (a-b)/(1-b) exactly, while both
     observers' marginal report rates stay calibrated (for = a, against = b). So the split is
     the same as in the agreement case: independent evidence -> problog a(1-b); completely
     coupled (one-on-one refutation) evidence -> mycin -- just as independent triggers give
     noisy-or and redundant triggers give max. MYCIN's conflict rule is the survival probability
     of a claim under paired refutation, not a rule about independent opposing observations.
     Caveat: the event here is "the claim survives scrutiny" (an evidence-level event), not the
     underlying fact itself. (Betting demonstration: decide_kelly_scenarios.py, world 6.)
"""
import os, sys, random, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib


def poisson(lam):
    """Knuth's Poisson sampler."""
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= random.random()
        if p <= L:
            return k - 1


def main(N=300_000, seed=1):
    random.seed(seed)

    print("=== Part A: mycin agreement case = noisy-or = at least one of two separate 'for' signs ===")
    print("  (strengths a,b of two separate 'for' observers; result strength = a+b-ab)")
    conf_for = lambda a: (1 + a) / 2
    for a, b in [(0.60, 0.80), (0.70, 0.70), (0.50, 0.90)]:
        hits = sum((random.random() < a) or (random.random() < b) for _ in range(N))
        freq = hits / N                                   # strength-scale frequency
        conf = poolib.mycin(conf_for(a), conf_for(b))     # named confidence value
        print(f"  a={a:.2f} b={b:.2f}: MC strength rate={freq:.4f}  a+b-ab={poolib.por(a, b):.4f}  "
              f"mycin conf={conf:.4f} -> strength {poolib.probtopneg(conf):.4f}  match")

    print("\n=== Part B: mycin conflict case = 1 - exp(-(lambda_a - lambda_b))  (an identity) ===")
    print("  (evidence rate lambda = -ln(1-x); opposing rates cancel)")
    for a, b in [(0.80, 0.30), (0.90, 0.60), (0.70, 0.20), (0.95, 0.50)]:
        la, lb = -math.log(1 - a), -math.log(1 - b)
        net_rate_value = 1 - math.exp(-(la - lb))
        print(f"  a={a:.2f} b={b:.2f}: mycin_diff={poolib.mycin_diff(a, b):.6f}  "
              f"1-exp(-(la-lb))={net_rate_value:.6f}  (identical)")

    print("\n=== Part C: is there an observation count for the conflict case? ===")
    print("  b1) sample one process at the already-cancelled net rate (la-lb): matches, but")
    print("      only because we put in the cancellation -- not a real combination of two observers.")
    print("  b2) two separate processes, net count = pro - con >= 1: does not match mycin.")
    print("  a ) separate pro and not con = a*(1-b) = problog:         does not match mycin.\n")
    for a, b in [(0.80, 0.30), (0.90, 0.60), (0.95, 0.50)]:
        la, lb = -math.log(1 - a), -math.log(1 - b)
        target = poolib.mycin_diff(a, b)
        # b1: direct net-rate process
        b1 = sum(poisson(la - lb) >= 1 for _ in range(N)) / N
        # b2: two independent processes, difference >= 1
        b2 = sum((poisson(la) - poisson(lb)) >= 1 for _ in range(N)) / N
        # a: independent pro and not con
        al = sum((random.random() < a) and not (random.random() < b) for _ in range(N)) / N
        print(f"  a={a:.2f} b={b:.2f}: mycin_diff={target:.4f} | "
              f"b1 net-rate={b1:.4f} (match) | b2 indep-diff={b2:.4f} (no) | a) a(1-b)={al:.4f} (no)")

    print("\n=== Part D: the coupling that DOES give a count -- one-on-one refutation ===")
    print("  for-signs ~ Poisson(la); each refuted independently w.p. lb/la (so refuted signs")
    print("  form the against-observer's Poisson(lb): marginals stay calibrated);")
    print("  event = at least one for-sign survives. P = 1-e^{-(la-lb)} = mycin_diff exactly.")
    for a, b in [(0.80, 0.30), (0.90, 0.60), (0.95, 0.50)]:
        la, lb = -math.log(1 - a), -math.log(1 - b)
        surv = agn = 0
        for _ in range(N):
            n_for = poisson(la)
            refuted = sum(random.random() < lb / la for _ in range(n_for))
            if n_for - refuted >= 1: surv += 1
            if refuted >= 1: agn += 1
        print(f"  a={a:.2f} b={b:.2f}: P(survivor)={surv/N:.4f}  mycin_diff={poolib.mycin_diff(a,b):.4f}"
              f"  (against marginal {agn/N:.4f} = b)  match")

    print("\n-> agreement: mycin matches a count (at least one of two separate 'for' signs).")
    print("-> conflict: under INDEPENDENT opposing evidence no natural count gives mycin (parts")
    print("   C-a, C-b2) -- there the right rules are problog a(1-b) or, for calibrated experts,")
    print("   upco (corr_upco_bayes.py). But under one-on-one REFUTATION (part D) mycin is exactly")
    print("   the survival probability of the claim: independent conflict -> problog, fully")
    print("   coupled conflict -> mycin, mirroring independent -> noisy-or vs redundant -> max.")
    print("   mycin's associativity then reads as rates adding (agreement) and cancelling")
    print("   (conflict) -- the uninorm additive generator with a physical meaning.")
    print("   Betting demonstration: decide_kelly_scenarios.py, worlds 6 vs 7.")


if __name__ == "__main__":
    main()
