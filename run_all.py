#!/usr/bin/env python3
"""Run the whole pooling-correspondence suite and print each result block."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import corr_averaging_mixture
import corr_max_dependent_or
import corr_problog_independent_or
import corr_upco_bayes
import corr_geometric_shared_evidence
import corr_mycin_probe
import corr_cumulation_dependence_spectrum
import decide_kelly_betting
import decide_cost_loss
import decide_kelly_scenarios
import corr_dempster_shafer
import decide_extremized_weighted_logodds
import decide_kelly_unknown_world
import decide_kelly_world_identification
try:
    import decide_murphy_diagram          # needs numpy + matplotlib; everything else is pure Python
except ImportError:
    decide_murphy_diagram = None


def sep(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


if __name__ == "__main__":
    sep("1. averaging  <->  two situations, only one of which is real (mixture)")
    corr_averaging_mixture.main()
    sep("2. max  <->  overlapping detectors   /   noisy-or  <->  separate detectors")
    corr_max_dependent_or.main()
    sep("3. problog  <->  two observers looking for signs separately")
    corr_problog_independent_or.main()
    sep("4. upco / product of experts  <->  two separate experts, even prior")
    corr_upco_bayes.main()
    sep("5. geometric  <->  two watchers who sampled the same thing")
    corr_geometric_shared_evidence.main()
    sep("6. mycin  <->  agreement: noisy-or; conflict: survival under refutation")
    corr_mycin_probe.main()
    sep("7. CONFER cumulation  <->  two proofs sharing part of their evidence")
    corr_cumulation_dependence_spectrum.main()
    sep("8. decision value: Kelly betting -- wealth growth per pooling rule")
    decide_kelly_betting.main()
    sep("9. decision value: cost-loss actions -- expense per pooling rule")
    decide_cost_loss.main()
    sep("10. every pool has its world: one betting scenario per pool where it wins")
    decide_kelly_scenarios.main()
    sep("11. Dempster-Shafer  <->  contradicting witnesses of known reliability")
    corr_dempster_shafer.main()
    sep("12. the log-odds family with a knob: extremized and weighted pools")
    decide_extremized_weighted_logodds.main()
    sep("13. betting when the world is unknown: one rule over a mixture of worlds")
    decide_kelly_unknown_world.main()
    sep("14. Murphy diagram: expense at every stake level + the Schervish identity check")
    if decide_murphy_diagram is None:
        print("skipped: needs numpy and matplotlib (pip install numpy matplotlib)")
    else:
        decide_murphy_diagram.main()
    sep("15. identifying the world by betting: a one-time fee of log2(K) bits")
    decide_kelly_world_identification.main()
