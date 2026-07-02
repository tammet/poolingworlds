#!/usr/bin/env python3
"""
max pooling         <->  at least one of two overlapping detectors fires.
noisy-or / problog  <->  at least one of two separate detectors fires.

Story. Two detectors each flag animals as birds. Detector 1 flags 30% of animals, detector 2
flags 80%. You call an animal a bird if at least one detector flags it. How often that happens
depends on how the two detectors relate.

  Separate detectors -- one checks the song, the other checks the beak. An animal missed by one
  can still be caught by the other, so the flag-rate is
        0.30 + 0.80 - 0.30*0.80 = 0.86        (noisy-or = problog).

  Overlapping detectors -- the second is a more sensitive copy of the first, so anything the
  first flags the second flags too. Then "at least one fires" is the same as "the more sensitive
  one fires", and the flag-rate is
        max(0.30, 0.80) = 0.80                (max).

This is the CONFER cumulative-confidence spectrum: two proofs of the same fact from separate
evidence give noisy-or; two proofs built from the same evidence give max; partial overlap sits
in between.

In code: the separate case gives each animal two independent random checks; the overlapping case
decides both detectors from one random number, so the weaker detector flags an animal only when
the stronger one also does.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib


def main(N=200_000, seed=1):
    random.seed(seed)
    print("=== max (overlapping detectors) vs noisy-or/problog (separate detectors) ===")
    print("Story: two bird-detectors flag rates p and q; you flag 'bird' if at least one fires.")
    print(f"(N={N} per pair)\n")
    pairs = [(0.30, 0.80), (0.20, 0.60), (0.50, 0.90), (0.40, 0.40), (0.10, 0.70)]
    names = ("avg", "max", "noisyor", "upco")
    for p, q in pairs:
        como = ind = 0
        for _ in range(N):
            u = random.random()
            if u < p or u < q:            # comonotonic: same u
                como += 1
            if (random.random() < p) or (random.random() < q):  # independent
                ind += 1
        f_como, f_ind = como / N, ind / N
        print(f"  overlapping " + poolib.fmt_compare(f_como, p, q, names))
        print(f"  separate    " + poolib.fmt_compare(f_ind, p, q, names))
        print()
    print("-> overlapping detectors match max; separate detectors match noisy-or/problog.")


if __name__ == "__main__":
    main()
