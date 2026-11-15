"""Appendix C: rolling selection with more precisely estimated betas.

Repeats the rolling-window part of 06_ch5_selection.py with each candidate
beta of the penalized methods taken from a regression on an intercept, the
market and that candidate alone, instead of the joint regression on the
market and all 61 candidates. Everything else is unchanged.

Run with BA_SCOPE=bivariate, then run 09_ch5_placebo.py in the same scope.
Writes P11_window_sets.csv, P2c_by_separation.csv, P2d_cluster_overlap.csv,
P11_out_of_sample.csv and P11_pairs.csv to output/bivariate/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from eufactors import config, rolling           # noqa: E402
from eufactors import selection as sel          # noqa: E402


def main():
    if config.CANDIDATE_BETAS != 'bivariate':
        raise SystemExit('run this script with BA_SCOPE=bivariate')
    U = sel.shared_universe_cut(config.COUNTRIES)
    p = len(U)
    samples = {c: sel.sample(c, universe=U, menu_cut=False) for c in config.COUNTRIES}
    ext = sel.selectors_extended()
    print('candidate betas: %s, candidates: %d' % (config.CANDIDATE_BETAS, p), flush=True)

    sets, _, starts = rolling.select_in_windows(samples, ext)
    rolling.window_sets_table(sets, starts).to_csv(config.out('P11_window_sets.csv'))
    rolling.overlap_by_separation(sets, p).to_csv(config.out('P2c_by_separation.csv'))
    rolling.cluster_overlap(samples, sets, ext).to_csv(config.out('P2d_cluster_overlap.csv'))
    summary, pairs = rolling.out_of_sample_r2(samples, sets, starts, list(ext))
    summary.to_csv(config.out('P11_out_of_sample.csv'))
    pairs.to_csv(config.out('P11_pairs.csv'))
    print('-> %s' % config.OUTPUT)


if __name__ == '__main__':
    main()
