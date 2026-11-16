"""Chapter 5: factor selection, stability and out-of-sample R^2.

Writes to output/<scope>/:
  P1_stability_frequencies.csv      selection frequency over 100 random halves, q = 8 (Tables 5, C1)
  P1_stability_frequencies_q4.csv   the same with q = 4
  P1b_null_band.csv                 highest frequency when no candidate is priced (Table C1, Panel C)
  P5_classics.csv                   frequencies of the classic characteristics and the base rate q/p (Table C1)
  P12_stable_sets.csv               stable sets and the Meinshausen-Buehlmann bound (Tables 4, C6, D2)
  P4_method_axis.csv                stable factors shared between methods (Table 4)
  P11_window_sets.csv               selected set per method and rolling window
  P2_time_axis.csv                  set sizes, empty windows, distinct factors (Table 4, Panel B)
  P2c_by_separation.csv             overlap relative to chance by separation (Figure 1, Tables C6, D2)
  P2d_cluster_overlap.csv           overlap of clusters of near substitutes (Table C3, Panel A)
  P7_vs_benchmark.csv               GRS and mean |alpha| of stable sets and benchmarks (Section 5.2.3)
  P11_out_of_sample.csv             out-of-sample R^2 (Tables 6, C6, D2)
  P11_pairs.csv                     the same per window pair (input to 09_ch5_placebo.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                       # noqa: E402
import pandas as pd                                      # noqa: E402

from eufactors import baseline, config, rolling          # noqa: E402
from eufactors import selection as sel                   # noqa: E402

CLASSICS = rolling.CLASSICS
PI = 0.60
N_DRAWS = 100
# stability settings of Section 5.1.3: (pi, q)
VARIANTS = {'pi0.60_q8': (0.60, 8), 'pi0.90_q8': (0.90, 8), 'pi0.60_q4': (0.60, 4)}


def save(name, df):
    df.to_csv(config.out(name + '.csv'))
    return df


def main():
    countries = config.COUNTRIES
    U = sel.shared_universe_cut(countries)
    p = len(U)
    print('candidate set: %d factors' % p, flush=True)
    samples = {c: sel.sample(c, universe=U, menu_cut=False) for c in countries}
    core = sel.selectors()
    ext = sel.selectors_extended()
    stab_sets = {8: dict(core, **{'ElasticNet a0.2': ext['ElasticNet a0.2']}),
                 4: sel.selectors(cap=4)}

    # --- stability selection ---------------------------------------------
    freqs = {}
    for q, sels_q in stab_sets.items():
        for c in countries:
            R, m, Z = samples[c]
            mk = m.to_numpy().ravel()
            for meth, fn in sels_q.items():
                rng = np.random.default_rng(sel.SEED)
                f, _ = sel.stability_selection(fn, R, mk, Z, n_draws=N_DRAWS, rng=rng)
                freqs[(c, meth, q)] = f
                print('  %s %-16s q=%d  max freq %.2f'
                      % (c, meth, q, f.max() if len(f) else 0.0), flush=True)

    def freq_frame(q):
        F_ = pd.DataFrame({'%s | %s' % (c, m_): v for (c, m_, qq), v in freqs.items()
                           if qq == q}).reindex(U).fillna(0.0)
        return F_.loc[F_.max(axis=1).sort_values(ascending=False).index]
    F = save('P1_stability_frequencies', freq_frame(8))
    save('P1_stability_frequencies_q4', freq_frame(4))

    # --- null band (Table C1, Panel C) --------------------------------------
    rows = []
    for c in countries:
        R, m, Z = samples[c]
        mk = m.to_numpy().ravel()
        for meth, fn in core.items():
            nmax = sel.null_frequencies(fn, R, mk, Z, n_draws=40,
                                        rng=np.random.default_rng(sel.SEED + 1))
            obs = float(F['%s | %s' % (c, meth)].max())
            rows.append(dict(country=c, method=meth, observed_max=obs,
                             null_max=nmax, clears_null=obs > nmax))
    save('P1b_null_band', pd.DataFrame(rows).set_index(['country', 'method']))

    # --- classic characteristics (Table C1, Panel B) ------------------------
    base = sel.CAP / float(p)
    cl = F.reindex(CLASSICS).fillna(0.0)
    cl_rel = cl / base
    cl_rel.columns = [c + ' (x base rate)' for c in cl.columns]
    save('P5_classics', pd.concat([cl, cl_rel], axis=1).assign(base_rate=base))

    # --- stable sets (Table 4, Panel A) and agreement between methods --------
    stable = {}
    rows = []
    for vname, (pi, q) in VARIANTS.items():
        for (c, meth, qq), f in freqs.items():
            if qq != q:
                continue
            s_ = set(f[f >= pi].index)
            stable[(c, meth, vname)] = s_
            rows.append(dict(variant=vname, pi=pi, q=q, country=c, method=meth,
                             K_stable=len(s_), members=', '.join(sorted(s_)),
                             mb_bound_EV=sel.mb_bound(q, pi, p)))
    save('P12_stable_sets', pd.DataFrame(rows).set_index(['variant', 'country', 'method']))

    rows4 = []
    for vname in VARIANTS:
        meths = sorted({m_ for (c, m_, v) in stable if v == vname})
        for c in countries:
            for i in range(len(meths)):
                for j in range(i + 1, len(meths)):
                    a = stable[(c, meths[i], vname)]
                    b = stable[(c, meths[j], vname)]
                    rows4.append(dict(variant=vname, country=c,
                                      pair='%s vs %s' % (meths[i], meths[j]),
                                      jaccard=sel.jaccard(a, b), shared=len(a & b)))
    save('P4_method_axis', pd.DataFrame(rows4).set_index(['variant', 'country', 'pair']))

    # --- rolling windows ------------------------------------------------------
    sets, capped, starts = rolling.select_in_windows(samples, ext)
    save('P11_window_sets', rolling.window_sets_table(sets, starts))

    rows = []
    for (c, meth), per in sets.items():
        js = [sel.jaccard(per[i], per[i + 1]) for i in range(len(per) - 1)]
        pairs = [(i, j) for i in range(len(per)) for j in range(i + 1, len(per))]
        allj = np.nanmean([sel.jaccard(per[i], per[j]) for i, j in pairs])
        alln = np.nanmean([sel.jaccard_random(len(per[i]), len(per[j]), p) for i, j in pairs])
        cps = [x for x in capped[(c, meth)] if x is not None]
        rows.append(dict(country=c, method=meth, n_windows=len(per),
                         empty_windows=sum(1 for x in per if not x),
                         mean_K=np.mean([len(x) for x in per]),
                         cap_binds_share=np.mean(cps) if cps else np.nan,
                         jaccard_consecutive=np.nanmean(js) if js else np.nan,
                         jaccard_all_pairs=allj,
                         all_pairs_x_chance=allj / alln if alln else np.nan,
                         distinct_factors_used=len(set().union(*per))))
    save('P2_time_axis', pd.DataFrame(rows).set_index(['country', 'method']))
    save('P2c_by_separation', rolling.overlap_by_separation(sets, p))
    save('P2d_cluster_overlap', rolling.cluster_overlap(samples, sets, ext))

    # --- GRS and mean |alpha| of the stable sets and benchmarks ---------------
    # Benchmark factors come from all JKP factors, so that a classic factor
    # that is not a candidate can still enter the six-factor model.
    rows, Z_bench = [], {}
    for c in countries:
        R, m, Z = samples[c]
        mk = m.to_numpy().ravel()
        Rv = R.to_numpy()
        Zfull = sel.zoo(c)
        Zfull.index = pd.PeriodIndex(Zfull.index, freq='M')
        Z = pd.concat([Z, Zfull.reindex(Z.index)[[k for k in CLASSICS if k not in Z.columns]]],
                      axis=1)
        Z_bench[c] = Z
        specs = {'CAPM': [], 'MKT + market_equity': ['market_equity'],
                 'MKT + at_gr1': ['at_gr1'], 'FF5+MOM': CLASSICS}
        seen = {tuple(v) for v in specs.values()}
        for (cc, meth, vname), s_ in stable.items():
            if cc == c and s_ and tuple(sorted(s_)) not in seen:
                specs['%s stable (%s)' % (meth, vname)] = sorted(s_)
                seen.add(tuple(sorted(s_)))
        for label, cols in specs.items():
            Fm = np.column_stack([mk] + ([Z[cols].to_numpy()] if cols else []))
            grs = baseline.grs_or_nan(Rv, Fm)
            _, t = sel.fmb(Rv, Fm)
            rows.append(dict(country=c, model=label, K=1 + len(cols),
                             members=', '.join(cols),
                             grs=grs['grs'], grs_p=grs['p_value'],
                             mean_abs_alpha_pct=grs['mean_abs_alpha_pct_mo'],
                             aa_over_ar=grs['aa_over_ar'],
                             n_sig_lambda_raw=int((np.abs(t[1:]) > 1.96).sum())))
    save('P7_vs_benchmark', pd.DataFrame(rows).set_index(['country', 'model']))

    # --- out-of-sample R^2 (Table 6) -------------------------------------------
    summary, pairs = rolling.out_of_sample_r2(samples, sets, starts,
                                              list(ext) + rolling.FIXED_MODELS,
                                              Z_bench=Z_bench)
    save('P11_out_of_sample', summary)
    save('P11_pairs', pairs)
    print('-> %s' % config.OUTPUT)


if __name__ == '__main__':
    main()
