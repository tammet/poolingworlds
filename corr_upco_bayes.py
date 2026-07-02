#!/usr/bin/env python3
"""
upco / multiplicative pooling / product of experts  <->  two experts who examined the animal
separately, when a bird and a non-bird are equally likely to start with.

Story. A caught animal is, before anyone looks, equally likely to be a bird or not. Two experts
examine it separately. Expert 1 is right 80% of the time, expert 2 is right 75%. Each gives the
probability that it is a bird, based on what they saw: when the 80%-expert says "bird" that
number is 0.80, when they say "not bird" it is 0.20. You now have two probabilities, p1 and p2,
and want one combined probability that it is a bird.

The experts looked at different things (say one listens to the song, the other inspects the
plumage), so what one sees tells you nothing extra about what the other sees once you know the
truth. Then the combined probability multiplies their odds:
      upco(p1,p2) = p1*p2 / ( p1*p2 + (1-p1)*(1-p2) ).
Two experts who both lean towards "bird" leave you more sure than either one alone.

This holds when a bird and a non-bird are equally likely to start with (prior 1/2). If birds are
rarer to begin with (say prior 0.30), each expert's number already contains that low starting
chance — multiplying the two numbers counts it twice, and upco comes out too high. The fix is to
divide the doubled starting odds back out once; that is the prior-corrected odds pool. This is
the 1/2-prior condition from memo section 10.6.

Model below: hidden truth H (bird or not); each expert sees a signal that matches the truth with
probability equal to their accuracy, the two signals independent given the truth; expert i
reports p_i = P(bird | their signal). We compare upco(p1,p2) with the full Bayes answer
P(bird | both signals) and with a Monte-Carlo count.
"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import poolib


def post(signal, acc, prior):
    lh = acc if signal == 1 else 1 - acc          # P(S| H=1)
    ln = (1 - acc) if signal == 1 else acc        # P(S| H=0)
    return lh * prior / (lh * prior + ln * (1 - prior))


def exact_joint(s1, s2, a1, a2, prior):
    def lik(s, a, h):
        return a if s == h else 1 - a
    num = lik(s1, a1, 1) * lik(s2, a2, 1) * prior
    den = num + lik(s1, a1, 0) * lik(s2, a2, 0) * (1 - prior)
    return num / den


def mc_joint(s1, s2, a1, a2, prior, N):
    hits = tot = 0
    for _ in range(N):
        H = 1 if random.random() < prior else 0
        S1 = H if random.random() < a1 else 1 - H
        S2 = H if random.random() < a2 else 1 - H
        if S1 == s1 and S2 == s2:
            tot += 1
            hits += H
    return hits / tot if tot else float("nan")


def run(prior, a1=0.80, a2=0.75, N=400_000, seed=1):
    random.seed(seed)
    print(f"prior={prior}  accuracies=({a1},{a2})  N={N}")
    print("  (S1,S2) | p1     p2    | MC freq  exactBayes | upco    bayes_pool | match")
    for s1 in (1, 0):
        for s2 in (1, 0):
            p1, p2 = post(s1, a1, prior), post(s2, a2, prior)
            fb = exact_joint(s1, s2, a1, a2, prior)
            fmc = mc_joint(s1, s2, a1, a2, prior, N)
            u = poolib.upco(p1, p2)
            bp = poolib.bayes_odds_pool([p1, p2], prior)
            tag = "upco" if abs(u - fb) < 1e-3 else ("bayes_pool" if abs(bp - fb) < 1e-3 else "-")
            print(f"   ({s1},{s2})  | {p1:.3f}  {p2:.3f} | {fmc:.4f}   {fb:.4f}     "
                  f"| {u:.4f}  {bp:.4f}     | {tag}")
    print()


def main():
    print("=== upco = Bayes over two separate experts, at prior 1/2 ===")
    print("Story: two experts examine the same animal separately and each gives P(bird).\n")
    run(prior=0.5)
    run(prior=0.3)
    print("-> at prior 1/2, upco equals the full Bayes answer and the Monte-Carlo count.")
    print("-> at prior other than 1/2, upco is off; the prior-corrected odds pool matches Bayes.")


if __name__ == "__main__":
    main()
