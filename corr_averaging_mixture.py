#!/usr/bin/env python3
"""
averaging (linear pool)  <->  a mixture of two situations, only one of which is the real one.

Story. Two habitats: scrubland and wetland. In scrubland 30% of the animals you catch are
birds; in wetland 80% are. You arrive at a site but do not know its habitat; it is equally
likely to be scrubland or wetland. The chance the next animal you catch is a bird is

      1/2 * 0.30  +  1/2 * 0.80  =  0.55,

the average of the two rates. Averaging is the right answer because the two rates describe two
separate situations and only one of them holds at this site. You are not combining two readings
of the same animal into a sharper estimate; you are accounting for not knowing which habitat you
are in. (Same maths, different dress: two people each measured the bird-rate in a different
region, and this case comes from region i with probability w_i.)

Model below: pick source i with probability w_i, then draw the event with probability p_i;
the long-run rate is sum w_i p_i.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib


def main(N=200_000, seed=1):
    random.seed(seed)
    print("=== averaging (linear pool) = a mixture of two situations ===")
    print("Story: two habitats -- scrubland bird-rate p1, wetland bird-rate p2; the site is")
    print("equally likely to be either, so the bird-rate is (p1+p2)/2.")
    print(f"(equal weights 0.5/0.5, N={N} per pair)\n")
    pairs = [(0.30, 0.80), (0.15, 0.65), (0.20, 0.50), (0.60, 0.90), (0.40, 0.70)]
    for p, q in pairs:
        hits = 0
        for _ in range(N):
            src = p if random.random() < 0.5 else q
            if random.random() < src:
                hits += 1
        freq = hits / N
        print("  " + poolib.fmt_compare(freq, p, q))
    print("\n-> avg matches the mixture rate; max, noisy-or, upco and geo do not.")


if __name__ == "__main__":
    main()
