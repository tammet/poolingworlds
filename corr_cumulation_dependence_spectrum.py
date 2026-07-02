#!/usr/bin/env python3
"""
CONFER cumulation: max at one end, noisy-or at the other, and where it is correct.

Story. You have two ways to conclude the animal is a bird. One argument uses the song recording
and the wing shape; another uses the wing shape and the beak photo. Each piece of evidence might
be wrong, so each argument on its own gives a confidence: c1 and c2. How sure are you now, with
both arguments in hand? It depends on the shared piece (the wing shape). If the arguments share
no evidence, the second one is a genuinely new chance of being right, and the combined confidence
is noisy-or, c1+c2-c1*c2. If they use exactly the same evidence, the second argument adds
nothing, and the combined confidence is just max(c1,c2). Partial sharing, as here, sits in
between.

Setting, formally. The same fact is derived by two proofs. Each proof, on its own, gives a
confidence (c1 and c2) that the fact holds. We want one combined confidence. CONFER's cumulation
rule (CADE-2021, Def. 6) is

    cumul(c1, c2, a) = max(c1 + c2*a, c1*a + c2) - c1*c2*a  =  max(c1,c2) + a*(noisy_or - max),

with a = i*h in [0,1].  a = 0 gives max(c1,c2); a = 1 gives noisy-or c1+c2-c1*c2.  So it slides
between the two along a straight line, and a is meant to be how independent the two proofs are.

What this is, in the literature.
  * max and noisy-or are two triangular conorms (t-conorms): max is the smallest t-conorm,
    noisy-or (probabilistic sum x+y-xy) is a larger one; see wiki concepts/TNorm.md and
    Klement, Mesiar & Pap, "Triangular Norms" (2000).  Parametric families such as Frank's
    interpolate continuously from max through probabilistic sum to the bounded sum.
  * The deeper fact is about dependence.  Given only the two marginals c1, c2, the probability
    P(A or B) is not fixed; it depends on how the two events relate.  The achievable range is
        max(c1,c2)  <=  P(A or B)  <=  min(1, c1+c2)
    (Frechet-Hoeffding / Boole bounds; see wiki concepts/PSAT.md, Hailperin, Nilsson).
        - lower end max      = maximal positive dependence (one event's occurrence is nested in
          the other's), i.e. the two proofs share all their luck;
        - independence       = noisy-or (strictly inside the range);
        - upper end min(1,·) = maximal negative dependence (the two proofs avoid each other).
    max is also the disjunctive fusion of possibility theory (wiki concepts/PossibilityTheory.md).

So CONFER's [max, noisy-or] is exactly the positive-dependence-to-independence half of the full
range.  This fits the use case: two proofs of the same fact can share evidence (pushing toward
max) or be independent (noisy-or), but they are not negatively dependent, so the value never
needs to go above noisy-or.

Below: three experiments.
  1. common-shock model -- the exact condition under which cumul is correct.
  2. shared-inputs model -- the faithful CONFER picture (endpoints exact, middle a heuristic).
  3. full dependence range -- max, noisy-or, bounded sum, with the negative-dependence case that
     falls outside CONFER's range.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib


def part1_common_shock(c1=0.50, c2=0.80, N=400_000, seed=1):
    """With probability lam the two proofs share the same luck (one random draw, giving max);
    otherwise they are independent (two draws, giving noisy-or).  Then
        P(A or B) = lam*max + (1-lam)*noisy-or = cumul(c1, c2, a=1-lam),  exactly."""
    random.seed(seed)
    print("=== 1. common-shock model: cumul is exactly correct here ===")
    print(f"c1={c1}, c2={c2}.  With prob lam the two proofs share one draw (-> max), else two")
    print(f"independent draws (-> noisy-or).  Expect union rate = cumul(c1,c2, a=1-lam).")
    print(f"max={poolib.mx(c1,c2):.4f}  noisy-or={poolib.noisy_or(c1,c2):.4f}  (N={N})\n")
    print("  lam   a=1-lam | union rate | cumul(c1,c2,a)")
    for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
        hits = 0
        for _ in range(N):
            if random.random() < lam:                 # shared luck: one draw
                u = random.random()
                a_ev, b_ev = u < c1, u < c2
            else:                                      # independent: two draws
                a_ev, b_ev = random.random() < c1, random.random() < c2
            if a_ev or b_ev:
                hits += 1
        a = 1 - lam
        print(f"  {lam:.2f}   {a:.2f}    |   {hits/N:.4f}   | {poolib.cumul(c1, c2, a):.4f}")
    print()


def part2_shared_inputs(m=4, q=0.85, N=400_000, seed=1):
    """Faithful CONFER picture.  Two proofs each need m evidence items (each valid w.p. q) to
    succeed; the two proofs share k of those items.  Fact holds if either proof succeeds.
      k=0  -> proofs independent -> noisy-or ;   k=m -> proofs identical -> max.
    CONFER estimates independence as i = 1 - shared/total = 1 - k/(2m-k) and predicts
    cumul(c1,c2,i).  Endpoints are exact; the middle is a heuristic."""
    random.seed(seed)
    c = q ** m                                          # c1 = c2 = c (symmetric)
    print("=== 2. shared-inputs model: faithful CONFER; endpoints exact, middle a heuristic ===")
    print(f"each proof needs m={m} items valid (each w.p. q={q}); proofs share k items.")
    print(f"c1=c2={c:.4f}   max={c:.4f}   noisy-or={poolib.noisy_or(c,c):.4f}  (N={N})\n")
    print("  k  shared   i=1-k/(2m-k) | union rate | cumul heuristic | note")
    for k in range(0, m + 1):
        total = 2 * m - k
        i = 1 - k / total
        hits = 0
        for _ in range(N):
            items = [random.random() < q for _ in range(total)]
            d1 = all(items[0:m])                        # proof 1 uses items 0..m-1
            d2 = all(items[m - k:m - k + m])            # proof 2 overlaps proof 1 in k items
            if d1 or d2:
                hits += 1
        note = "independent -> noisy-or" if k == 0 else ("identical -> max" if k == m else "")
        print(f"  {k}   {k}/{total}    {i:.3f}      |   {hits/N:.4f}   |     {poolib.cumul(c, c, i):.4f}      | {note}")
    print()


def part3_full_range(c1=0.50, c2=0.80, N=400_000, seed=1):
    """The three landmark dependence structures and the full achievable range for P(A or B)."""
    random.seed(seed)
    print("=== 3. full dependence range for P(A or B) (Frechet-Hoeffding / Boole) ===")
    print(f"c1={c1}, c2={c2}\n")
    # comonotone: one draw, nested
    como = sum((lambda u: (u < c1) or (u < c2))(random.random()) for _ in range(N)) / N
    # independent: two draws
    indep = sum((random.random() < c1) or (random.random() < c2) for _ in range(N)) / N
    # countermonotone: one draw, events pushed apart
    counter = sum((lambda u: (u < c1) or (u >= 1 - c2))(random.random()) for _ in range(N)) / N
    print(f"  positive dep. (comonotone)   union rate = {como:.4f}   formula max         = {poolib.mx(c1,c2):.4f}")
    print(f"  independent                  union rate = {indep:.4f}   formula noisy-or    = {poolib.noisy_or(c1,c2):.4f}")
    print(f"  negative dep. (countermono)  union rate = {counter:.4f}   formula bounded sum = {poolib.boundedsum(c1,c2):.4f}")
    print(f"\n  achievable range = [max={poolib.mx(c1,c2):.4f} , bounded-sum={poolib.boundedsum(c1,c2):.4f}]")
    print("  CONFER cumulation covers [max , noisy-or] -- positive dependence up to independence.")
    print("  Going above noisy-or needs negative dependence, which two proofs of one fact do not have.")


def main():
    part1_common_shock()
    part2_shared_inputs()
    part3_full_range()


if __name__ == "__main__":
    main()
