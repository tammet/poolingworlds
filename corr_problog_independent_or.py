#!/usr/bin/env python3
"""
problog combinator  <->  two separate pieces of evidence about the same animal.

problog and mycin work on the confidence scale, where 0.5 means "no information". A confidence
maps to an evidence strength a = 2*conf - 1: for example, confidence 0.9 that it is a bird
means strength 0.8, read here as "a sign that it is a bird turns up in 80% of such cases";
confidence 0.5 means strength 0, no sign at all. The simulation runs on that strength scale.

Story.
  Two observers, both looking for signs it is a bird. Observer 1 finds a sign in 30% of cases,
  observer 2 in 80%. They look separately. You say "bird" if either finds a sign:
        0.30 + 0.80 - 0.30*0.80 = 0.86        (noisy-or).

  One observer looks for signs it is a bird (finds one 80% of the time), another looks for signs
  it is not a bird (finds one 30% of the time), separately. You count it as support for "bird"
  when the first finds a sign and the second does not:
        0.80 * (1 - 0.30) = 0.56.

The named confidence is pnegtoprob(strength) = problog(conf_for(a), conf_*(b)), with
conf_for(a) = (1+a)/2 (a "for" confidence, at least 0.5) and conf_against(b) = (1-b)/2 (an
"against" confidence, at most 0.5). This is the rule ProbLog uses, and the CONFER-2021 naive
combinator on the agreement case.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib

conf_for     = lambda a: (1 + a) / 2      # positive proof strength a -> confidence >= 0.5
conf_against = lambda b: (1 - b) / 2      # negative proof strength b -> confidence <= 0.5


def main(N=200_000, seed=1):
    random.seed(seed)
    print("=== problog = separate-evidence rate (strength scale) ===")
    print("Story: two observers look for signs about the same animal, separately.")
    print(f"(N={N} per pair)\n")

    print("Agreement (two 'for' observers, strengths a,b): chance at least one finds a sign = a+b-ab")
    print("  a    b   | MC strength freq | a+b-ab  | problog conf | strength(problog)")
    for a, b in [(0.30, 0.80), (0.50, 0.50), (0.60, 0.90), (0.20, 0.20)]:
        hits = sum((random.random() < a) or (random.random() < b) for _ in range(N))
        freq = hits / N
        conf = poolib.problog(conf_for(a), conf_for(b))     # both confidences >= 0.5
        print(f"  {a:.2f} {b:.2f} |     {freq:.4f}       | {poolib.por(a, b):.4f}  |   {conf:.4f}    |     {poolib.probtopneg(conf):.4f}")

    print("\nConflict ('for' observer a vs 'against' observer b): for finds a sign, against does not = a*(1-b)")
    print("  a    b   | MC strength freq | a(1-b)  | problog conf | strength(problog)")
    for a, b in [(0.80, 0.30), (0.90, 0.60), (0.70, 0.70), (0.60, 0.20)]:
        hits = sum((random.random() < a) and not (random.random() < b) for _ in range(N))
        freq = hits / N
        conf = poolib.problog(conf_for(a), conf_against(b))  # one >=0.5, one <=0.5
        print(f"  {a:.2f} {b:.2f} |     {freq:.4f}       | {a * (1 - b):.4f}  |   {conf:.4f}    |     {poolib.probtopneg(conf):.4f}")

    print("\n-> both problog cases match the separate-evidence rate.")


if __name__ == "__main__":
    main()
