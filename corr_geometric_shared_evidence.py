#!/usr/bin/env python3
"""
geometric / log-linear pooling  <->  two watchers who studied the same thing and now have
overlapping information (Pettigrew & Weisberg, "Geometric Pooling", Prop. 4).

Story. A species sings with some unknown rate. Two birdwatchers each watch a sample of this
species. Watcher 1 hears 12 of 20 sing, watcher 2 hears 18 of 20. From their samples each ends
up with a belief about the rate -- a probability curve over the possible rates. Those two curves
are the two confidences here. You want one combined belief.

When both watchers studied the same species, the right combination is the geometric pool of the
two curves. It gives the same belief you would get by putting their counts together as one sample
and halving it: 15 of 20 (that is, (12+18)/2 of (20+20)/2). Why halve? If the watchers had
genuinely separate information you would add their samples, 30 of 40 — that stronger combination
is what multiplying the curves (the upco case) gives. But here you cannot tell how much of their
information is the same information seen twice, so the geometric pool plays it safe: it keeps the
combined direction but only the weight of an average-sized sample. Combining the two curves by
plain averaging (the linear pool) is wrong in a different way: it gives a belief with two separate
humps, one at each watcher's rate, as if exactly one of them were right.

So the three rules line up by how much the watchers' information overlaps: fully separate ->
multiply (upco, sample sizes add); same kind of evidence, overlap unknown -> geometric (sample
sizes average); only one watcher is right -> linear (a mixture, as in the averaging experiment).
The match here is between whole belief curves (largest gap across a grid of rates), not a single
event rate; the simulation is used only to generate the two watchers' counts.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib

GRID = [i / 1000 for i in range(1, 1000)]        # theta in (0,1)


def normalize(vals):
    s = sum(vals)
    return [v / s for v in vals]


def binom_posterior(k, n):                        # unnormalized theta^k (1-theta)^(n-k), then normalize
    return normalize([(t ** k) * ((1 - t) ** (n - k)) for t in GRID])


def geo_pool(p1, p2, w=0.5):
    return normalize([(a ** w) * (b ** w) for a, b in zip(p1, p2)])


def lin_pool(p1, p2, w=0.5):
    return [w * a + (1 - w) * b for a, b in zip(p1, p2)]


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def main(seed=1):
    random.seed(seed)
    print("=== geometric pooling = belief on the combined (shared) sample ===")
    print("Story: two birdwatchers sample the same singing-rate; merge their belief curves.\n")
    true_theta = 0.7
    for (n1, n2) in [(20, 20), (40, 40), (30, 30)]:
        k1 = sum(random.random() < true_theta for _ in range(n1))
        k2 = sum(random.random() < true_theta for _ in range(n2))
        post1, post2 = binom_posterior(k1, n1), binom_posterior(k2, n2)
        gpool = geo_pool(post1, post2)
        lpool = normalize(lin_pool(post1, post2))
        # averaged sample posterior: (k1+k2)/2 heads in (n1+n2)/2 flips (fractional counts ok)
        kbar, nbar = (k1 + k2) / 2, (n1 + n2) / 2
        avg_post = normalize([(t ** kbar) * ((1 - t) ** (nbar - kbar)) for t in GRID])
        def verdict(d):
            return "match" if d < 1e-9 else "no match"
        dg, dl = maxdiff(gpool, avg_post), maxdiff(lpool, avg_post)
        print(f"  true theta={true_theta}, data: {k1}/{n1} and {k2}/{n2}")
        print(f"    max|geo_pool - combined_sample_belief|    = {dg:.2e}  ({verdict(dg)})")
        print(f"    max|linear_pool - combined_sample_belief| = {dl:.2e}  ({verdict(dl)})")
        if (k1, n1) == (k2, n2):
            print("    (the two watchers happened to get identical counts, so their belief curves")
            print("     are identical and every pooling rule trivially returns that same curve)")
    print("\n-> geometric pooling always gives the combined-sample belief; linear pooling only")
    print("   when the two samples happen to coincide.")


if __name__ == "__main__":
    main()
