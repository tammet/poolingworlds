#!/usr/bin/env python3
"""
Shared pooling functions + Monte-Carlo helpers for the "pooling correspondence" suite.

Probability scale [0,1]; 0.5 = "no information". Signed scale [-1,0,1] via
probtopneg/pnegtoprob.  The mycin/upco/problog implementations are kept identical
to the ones used in the authors' earlier exploratory experiments on purpose.

Each pooling function corresponds to a different story about how the two confidences
arise; the corr_*.py scripts each realize one story as a Monte-Carlo rate and show
which pooling function matches it.  Summary map:

  averaging (linear)      <->  two situations, only one of which is real (mixture)
  max                     <->  at least one of two overlapping/redundant detectors fires
  problog / noisy-or      <->  at least one of two separate detectors/observers fires
  upco / mult / PoE       <->  two experts who examined the case separately, even prior
  geometric / log-linear  <->  two watchers who sampled the same thing (shared evidence)
  mycin                   <->  agreement = noisy-or; conflict = cancel opposing evidence
                               rate-for-rate; exact frequency under one-on-one refutation
                               (see corr_mycin_probe.py)
  CONFER cumulation       <->  two proofs sharing part of their evidence: slides from
                               max (all shared) to noisy-or (none shared)
                               (see corr_cumulation_dependence_spectrum.py)
  Dempster-Shafer         <->  contradicting witnesses of known reliability, who cannot
                               both be reliable; bel/pl = forced/not-excluded counts,
                               pignistic BetP = the betting value
                               (see corr_dempster_shafer.py)
  weighted/extremized     <->  reports with distorted log-odds: symmetric distortion k ->
  log-odds pools               alpha = 1/k (alpha > 1 = extremizing); per-expert k_i ->
                               weights 1/k_i; weighted linear <-> unequal mixture
                               (see decide_extremized_weighted_logodds.py)
"""

import math

# ---------------------------------------------------------------- scale maps
def probtopneg(x):  return x * 2 - 1          # 0..1     -> -1..0..1
def pnegtoprob(x):  return (x + 1) / 2        # -1..0..1 -> 0..1
def neg(x):         return -x
def ab(x):          return x if x >= 0 else -x
def near0(x):       return 0 <= x < 1e-9
def por(x, y):      return x + y - x * y      # probabilistic OR = noisy-or

# ---------------------------------------------------------------- pooling fns
def avg(p, q):                                # linear / arithmetic pool
    return (p + q) / 2

def mx(p, q):                                 # max pool
    return max(p, q)

def noisy_or(p, q):                           # = problog agreement case
    return por(p, q)

def upco(p, q):                               # multiplicative / product-of-experts
    if round(p, 6) == 0 and round(q, 6) == 1: return 0.5
    if round(p, 6) == 1 and round(q, 6) == 0: return 0.5
    d = p * q + (1 - p) * (1 - q)
    return 0.5 if d == 0 else (p * q) / d

def geo(p, q, w=0.5):                          # geometric / log-linear pool (binary)
    a = (p ** w) * (q ** w)
    b = ((1 - p) ** w) * ((1 - q) ** w)
    return 0.5 if a + b == 0 else a / (a + b)

def boundedsum(p, q):                          # Lukasiewicz t-conorm; upper Frechet bound for P(A or B)
    return min(1.0, p + q)

def cumul(c1, c2, a):
    """CONFER cumulative confidence (CADE-2021, Def. 6), with a = i*h in [0,1].
    Equals max(c1,c2) + a*(noisy_or - max):  a=0 -> max (fully dependent),
    a=1 -> noisy-or (independent).  A one-parameter interpolation between the two
    t-conorms max and probabilistic sum."""
    return max(c1 + c2 * a, c1 * a + c2) - c1 * c2 * a

# ------------------------------------------------- Dempster-Shafer (binary frame, conflict)
# One-sided masses: m1 supports the claim, m2 opposes it, the rest of each is ignorance.
# Dempster's rule discards the contradictory mass m1*m2 and renormalizes. Agreement case
# (both masses on the same side) is plain noisy-or -- no conflict, no renormalization.
def ds_bel(m1, m2):                            # lower probability of the claim
    n = 1 - m1 * m2
    return (m1 * (1 - m2)) / n if n > 0 else 0.5

def ds_pl(m1, m2):                             # upper probability of the claim
    n = 1 - m1 * m2
    return 1 - (m2 * (1 - m1)) / n if n > 0 else 0.5

def ds_pignistic(m1, m2):                      # Smets' betting probability: bel + half the ignorance
    n = 1 - m1 * m2
    if n <= 0: return 0.5
    return (m1 * (1 - m2) + 0.5 * (1 - m1) * (1 - m2)) / n

# ------------------------------------------------- weighted / extremized log-odds pools
import math as _math

def _logit(p):   return _math.log(p / (1 - p))
def _sigm(x):    return 1 / (1 + _math.exp(-x))

def wlogodds(p, q, w1, w2):
    """General log-odds pool: pooled log-odds = w1*logit(p) + w2*logit(q).
    (w1,w2)=(0.5,0.5) -> geo;  (1,1) -> upco;  w1=w2=alpha -> the extremized M^alpha family."""
    p = min(max(p, 1e-12), 1 - 1e-12); q = min(max(q, 1e-12), 1 - 1e-12)
    return _sigm(w1 * _logit(p) + w2 * _logit(q))

def alpha_logodds(p, q, a):
    """Symmetric extremized pool: a=0.5 geo, a=1 upco, a>1 extremizing."""
    return wlogodds(p, q, a, a)

def wavg(p, q, w):
    """Weighted linear pool w*p + (1-w)*q (weights = situation probabilities in a mixture)."""
    return w * p + (1 - w) * q

# MYCIN certainty-factor combination, in the REVISED form used by later MYCIN/EMYCIN
# (van Melle; documented in Buchanan & Shortliffe, Rule-Based Expert Systems, 1984):
# on signed strengths x, y in [-1, 1], agreement combines as x + y - xy (noisy-or of
# magnitudes) and conflict as (x + y) / (1 - min(|x|, |y|)).  The original 1975 rule
# (Shortliffe & Buchanan, Math. Biosci. 23) added opposite-sign factors WITHOUT the
# renormalizing denominator.  mycin() below is the same rule carried to the confidence
# scale [0, 1] with 0.5 = "no information", via strength = 2*conf - 1.
def mycin_diff(x, y):                          # conflict case on strength magnitudes
    if x == y:   return 0.0
    if x > y:    return (x - y) / (1 - y)
    return (y - x) / (1 - x)

def mycin(x, y):                               # revised MYCIN rule on the confidence scale
    if (x == 0 and y == 1) or (x == 1 and y == 0): return 0.5
    x1 = ab(probtopneg(x)); y1 = ab(probtopneg(y))
    if x >= 0.5 and y >= 0.5:
        z = por(x1, y1)
    elif x <= 0.5 and y <= 0.5:
        z = neg(por(x1, y1))
    elif x >= 0.5 and y <= 0.5:
        if x == 1 and near0(y):     return 0.0
        if not near0(y) and x == 1: return 1.0
        if near0(y) and x != 1:     return 0.0
        z = mycin_diff(x1, y1)
        if y1 > x1: z = neg(z)
    else:  # x <= 0.5 and y >= 0.5
        if near0(x) and y == 1:     return 0.0
        if not near0(x) and y == 1: return 1.0
        if near0(x) and y != 1:     return 0.0
        z = mycin_diff(x1, y1)
        if x1 > y1: z = neg(z)
    return pnegtoprob(z)

def problog(x, y):
    x1 = ab(probtopneg(x)); y1 = ab(probtopneg(y))
    if x <= 0.5 and y <= 0.5:
        z = neg(por(x1, y1))
    elif x >= 0.5 and y >= 0.5:
        z = por(x1, y1)
    else:                                      # different polarities
        if x <= 0.5 and y >= 0.5: z = y1 * (1 - x1)
        else:                     z = x1 * (1 - y1)
    return pnegtoprob(z)

# ---------------------------------------------------------------- Bayes helpers
def odds(p):
    return float("inf") if p >= 1 else p / (1 - p)

def prob_from_odds(o):
    return 1.0 if o == float("inf") else o / (1 + o)

def bayes_odds_pool(ps, prior=0.5):
    """Prior-corrected product-of-odds pool: posterior odds = prod(o_i)/O0^(n-1).
    Equals upco exactly when prior == 0.5.  Treats each p_i as a posterior of the
    same shared prior over conditionally-independent evidence."""
    O0 = prior / (1 - prior)
    prod_o = 1.0
    for p in ps:
        prod_o *= odds(p)
    O = prod_o / (O0 ** (len(ps) - 1))
    return prob_from_odds(O)

# ---------------------------------------------------------------- reporting
POOLS = {                                      # name -> 2-arg pooling fn
    "avg":     avg,
    "max":     mx,
    "problog": problog,
    "mycin":   mycin,
    "upco":    upco,
    "geo":     geo,
    "noisyor": noisy_or,
}

# Two scales (see notes.tex):
#  * PROBABILITY-scale opinion pools -- inputs are probabilities/credences in [0,1]:
#       avg, max, noisyor, upco, geo          (compared in the Group-1 experiments)
#  * CONFIDENCE combinators (CONFER/GK) -- inputs are confidences, 0.5 = "no info",
#    internally = signed strength a = 2*conf-1:  problog, mycin
#    (their correspondence is on the STRENGTH scale; see corr_problog / corr_mycin_probe).
PROB_SCALE_NAMES = ("avg", "max", "noisyor", "upco", "geo")

def compare_row(freq, p, q, names=PROB_SCALE_NAMES):
    """Return dict name->(value, abs error vs freq) and the best-matching name(s).
    Functions whose values coincide (a tie) are all reported, joined with '='."""
    out = {}
    for n in names:
        v = POOLS[n](p, q)
        out[n] = (v, abs(v - freq))
    best_err = min(out[n][1] for n in names)
    best_val = next(out[n][0] for n in names if out[n][1] == best_err)
    tied = [n for n in names if abs(out[n][0] - best_val) < 1e-9]
    return out, "=".join(tied)

def fmt_compare(freq, p, q, names=PROB_SCALE_NAMES):
    out, best = compare_row(freq, p, q, names)
    cells = "  ".join(f"{n}={out[n][0]:.4f}" for n in names)
    err = min(out[n][1] for n in names)
    return f"p=({p:.2f},{q:.2f}) freq={freq:.4f} | {cells} | best={best} (err {err:.4f})"
