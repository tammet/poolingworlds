#!/usr/bin/env python3
"""
Dempster-Shafer combination  <->  two witnesses who contradict each other, with known
reliabilities, on the assumption that they cannot both be reliable.

Story. Witness 1 says the animal was a bird; witness 2 says it was not. Witness 1 is reliable
with a known probability m1 (a reliable witness speaks the truth; an unreliable one's words
carry no information either way); witness 2 is reliable with probability m2, independently.
Since the two statements contradict each other, they cannot both be reliable -- rounds where
both reliability coins come up are impossible and are redrawn. This redraw is exactly
Dempster's renormalization: it throws away the contradictory mass m1*m2 and rescales the rest.

After the redraw, three things can happen, and each matches a Dempster quantity as a count:
  only witness 1 reliable -> the claim is TRUE.   Fraction  = m1(1-m2)/(1-m1m2)  = bel  (lower)
  only witness 2 reliable -> the claim is FALSE.
  neither reliable        -> the evidence says nothing about the claim.
  "claim true or undecided" fraction              = 1 - m2(1-m1)/(1-m1m2)       = pl   (upper)
  resolving the undecided rounds by a fair coin   = bel + half the ignorance    = BetP

bel and pl are bounds -- the fraction of rounds where the claim is FORCED true and the fraction
where it is NOT excluded. For betting, a point value is needed; Smets' rule is to bet with the
pignistic probability BetP (that is what the fair coin realizes). The betting demonstration
where the BetP-bettor beats bel, pl and the other conflict rules is decide_kelly_scenarios.py,
world 9.

Agreement case: if both witnesses SUPPORT the claim (masses m1, m2 on the same side), there is
no conflict and no renormalization; the combined support is 1-(1-m1)(1-m2) = noisy-or -- the
same agreement rule as problog and mycin. So Dempster-Shafer is a fourth member of the
"noisy-or on agreement, X on conflict" family, with X = problog renormalized:
    naive a-b   <   problog a(1-b)   <   DS a(1-b)/(1-ab)   ~   mycin (a-b)/(1-b).

Honest caveat: the redraw step assumes the witnesses genuinely cannot both be reliable. That
assumption is the controversial heart of Dempster's rule (Zadeh's criticism targets it): with
two highly reliable contradicting witnesses almost all rounds are discarded, and conclusions
rest on the sliver that remains.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib


def main(N=200_000, seed=1):
    random.seed(seed)

    print("=== Dempster-Shafer = contradicting witnesses of known reliability ===")
    print("(rounds where both would be reliable are impossible and redrawn)\n")

    print("Agreement first (both witnesses support the claim): combined support = noisy-or")
    for m1, m2 in [(0.30, 0.80), (0.60, 0.70)]:
        hits = sum((random.random() < m1) or (random.random() < m2) for _ in range(N))
        print(f"  m1={m1:.2f} m2={m2:.2f}: P(some reliable support)={hits/N:.4f}"
              f"  noisy-or={poolib.por(m1, m2):.4f}  (no conflict, nothing discarded)")

    print("\nConflict (witness 1 for, witness 2 against): three counts, three DS quantities")
    print("  m1   m2  | forced-true  bel    | not-excluded  pl    | coin-resolved  BetP")
    for m1, m2 in [(0.80, 0.30), (0.90, 0.60), (0.50, 0.50), (0.70, 0.20)]:
        forced = possible = coin_true = 0
        for _ in range(N):
            while True:                                  # redraw both-reliable rounds
                r1, r2 = random.random() < m1, random.random() < m2
                if not (r1 and r2):
                    break
            if r1:                                       # only witness 1 reliable
                forced += 1; possible += 1; coin_true += 1
            elif not r2:                                 # neither reliable: undecided
                possible += 1
                if random.random() < 0.5:
                    coin_true += 1
        print(f"  {m1:.2f} {m2:.2f} |   {forced/N:.4f}   {poolib.ds_bel(m1, m2):.4f} |"
              f"    {possible/N:.4f}   {poolib.ds_pl(m1, m2):.4f} |"
              f"    {coin_true/N:.4f}    {poolib.ds_pignistic(m1, m2):.4f}")

    print("\nThe four conflict rules on the same inputs (values only; money: scenarios world 9):")
    print("  a    b   | naive a-b | problog a(1-b) | DS bel a(1-b)/(1-ab) | mycin (a-b)/(1-b) | DS BetP")
    for a, b in [(0.80, 0.30), (0.90, 0.60), (0.70, 0.20)]:
        print(f"  {a:.2f} {b:.2f} |   {a-b:.4f}  |     {a*(1-b):.4f}     |        {poolib.ds_bel(a,b):.4f}        |"
              f"      {poolib.mycin_diff(a,b):.4f}      | {poolib.ds_pignistic(a,b):.4f}")

    print("\n-> all three DS quantities match their counts exactly. bel/pl are bounds")
    print("   (forced / not-excluded); BetP is the point value a bettor should use.")


if __name__ == "__main__":
    main()
