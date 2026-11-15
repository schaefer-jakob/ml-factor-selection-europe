"""Chapter 5: coverage, dense benchmarks, SDF shrinkage and the lucky-factors
bootstrap.

  G4a/G4b  test-asset coverage and the balanced panel per market (Section 3.2)
  G4c      size of the shared candidate set (Section 5.1.1)
  G1       PC1-PC8 against random dense and random named sets (out-of-sample
           R^2; not reported, see below)
  G2a      SDF shrinkage of Kozak et al. (2020), cross-validated R^2 on the
           full sample (Table C5, Panel A)
  G2b      the same out of sample, window w to w + s (Table C5, Panel B)
  G2c      edges between the SDF variants (not reported)
  G3a      lucky-factors bootstrap on the full sample (Tables 4 and C2)
  G3b      lucky-factors selection in every rolling window (input to
           10_ch5_sharpe.py)

G1 and G2c are not reported in the thesis. They are computed because G1, G2
and the random draws of G2c share one random-number stream, so dropping them
would change the random benchmark of Table C5, Panel B.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402

from eufactors import config, coverage, fmb, kns, pca, rolling   # noqa: E402
from eufactors import dense_benchmark as db                      # noqa: E402
from eufactors import lucky_factors as lf                        # noqa: E402
from eufactors import selection as sel                           # noqa: E402

SEPS = rolling.SEPARATIONS_OOS
KAPPAS_ROLL = np.geomspace(0.01, 10.0, 13)      # coarser grid inside windows
N_RANDOM_KNS = 20
N_BOOT_ROLLING = 200
COVERAGE_MARKETS = ('DEU', 'GBR', 'FRA', 'CHE', 'SWE')


def save(name, df):
    df.to_csv(config.out(name + '.csv'))
    print('  -> %s.csv (%d rows)' % (name, len(df)), flush=True)
    return df


def edge_row(**kw):
    """Mean edge, i.i.d. and Newey-West t-statistics, windows won."""
    d = np.asarray(kw.pop('d'), float)
    se_iid = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    se_nw = fmb.newey_west_se(d)
    kw.update(n_pairs=len(d), edge=d.mean(), t_iid=d.mean() / se_iid,
              t_nw=d.mean() / se_nw, won=int((d > 0).sum()))
    return kw


def load(countries):
    U = sel.shared_universe_cut(tuple(countries))
    print('candidate set (%s): %d factors' % ('+'.join(countries), len(U)), flush=True)
    samples = {c: sel.sample(c, universe=U, menu_cut=False) for c in countries}
    starts = {c: rolling.window_starts(len(samples[c][0])) for c in countries}
    return U, samples, starts


def pairs(st, sep):
    for i in range(len(st) - sep):
        yield (i, slice(st[i], st[i] + sel.WINDOW),
               slice(st[i + sep], st[i + sep] + sel.WINDOW))


# ------------------------------------------------------------- coverage ---
def scope_coverage():
    print('\ncoverage', flush=True)
    cov = pd.concat([coverage.leg_coverage(c) for c in COVERAGE_MARKETS])
    save('G4a_leg_coverage', cov.set_index(['country', 'characteristic']))
    summ = cov.groupby('country').agg(
        chars=('characteristic', 'size'), present=('present', 'sum'),
        complete_95=('complete_share', lambda s: int((s >= 0.95).sum())),
        worst_share=('complete_share', 'min'),
        latest_first_complete=('first_complete', 'max'))
    spans = []
    for c in COVERAGE_MARKETS:
        try:
            spans.append(coverage.panel_span(c))
        except Exception as e:
            spans.append(dict(country=c, note='%s: %s' % (type(e).__name__, e)))
    spans = pd.DataFrame(spans).set_index('country')
    save('G4b_panel_span', spans.join(summ))
    menus = coverage.menu_sizes([['DEU', 'GBR'], ['DEU', 'GBR', 'FRA'], ['FRA'], ['CHE'],
                                 ['SWE'], ['CHE', 'SWE'], ['DEU', 'GBR', 'FRA', 'CHE', 'SWE']])
    save('G4c_candidate_menu', menus.set_index('countries'))


# --------------------------------------------------------- dense benchmark ---
def dense_benchmark(samples, starts, rng):
    print('\nG1 dense random benchmark', flush=True)
    k = sel.CAP
    rows, prs = [], []
    for c, (R, m, Z) in samples.items():
        mk = m.to_numpy().ravel()
        cols = list(Z.columns)
        p = len(cols)
        for sep in SEPS:
            rec = {key: [] for key in ('pc18', 'pc916', 'rdense', 'rnamed')}
            for i, a_, b_ in pairs(starts[c], sep):
                Za, Zb = Z.iloc[a_], Z.iloc[b_]
                V = pca.pcs(Za)[0]
                Rb = R.iloc[b_]

                def oos(Vk):
                    return rolling.csr2_F(Rb, np.column_stack([mk[b_], db.project(Za, Zb, Vk)]))
                r_pc = oos(V[:, :k])
                r_916 = oos(V[:, k:2 * k])
                r_dense = np.mean([oos(db.random_directions(p, k, rng))
                                   for _ in range(db.N_RANDOM)])
                r_named = np.mean([rolling.csr2(Rb, mk[b_], Zb, rng.choice(cols, k, replace=False))
                                   for _ in range(db.N_RANDOM)])
                for key, v in zip(rec, (r_pc, r_916, r_dense, r_named)):
                    rec[key].append(v)
                prs.append(dict(country=c, separation=sep, window=i, PC1_8=r_pc,
                                PC9_16=r_916, random_dense=r_dense, random_named=r_named))
            A = {key: np.array(v) for key, v in rec.items()}
            shared = max(0, sel.WINDOW - sep * rolling.STEP)
            for label, x, y in (('PC1-8 vs random dense', 'pc18', 'rdense'),
                                ('PC1-8 vs random named', 'pc18', 'rnamed'),
                                ('PC1-8 vs PC9-16', 'pc18', 'pc916'),
                                ('random dense vs random named', 'rdense', 'rnamed'),
                                ('PC9-16 vs random dense', 'pc916', 'rdense')):
                rows.append(edge_row(country=c, separation=sep, months_shared=shared,
                                     comparison=label, R2_a=A[x].mean(), R2_b=A[y].mean(),
                                     d=A[x] - A[y]))
            print('  %s s=%d  PC1-8 %.3f  random dense %.3f  random named %.3f  PC9-16 %.3f'
                  % (c, sep, A['pc18'].mean(), A['rdense'].mean(), A['rnamed'].mean(),
                     A['pc916'].mean()), flush=True)
    save('G1_dense_benchmark', pd.DataFrame(rows).set_index(['country', 'separation', 'comparison']))
    save('G1_pairs', pd.DataFrame(prs).set_index(['country', 'separation', 'window']))


# ------------------------------------------------------------ SDF shrinkage ---
def sdf_shrinkage(samples, starts, rng):
    print('\nG2 SDF shrinkage', flush=True)
    rows = []
    for c, (R, m, Z) in samples.items():
        F, _, _ = kns.prepare(Z.to_numpy(), m.to_numpy().ravel())
        for space, rot in (('raw', False), ('PC', True)):
            surf, cols = kns.cv_surface(F, rotate_pc=rot)
            for i, kap in enumerate(kns.KAPPAS):
                for j, col in enumerate(cols):
                    rows.append(dict(country=c, space=space, kappa=kap,
                                     nonzero=col, cv_R2=surf[i, j]))
            print('  %s %-3s full-sample CV R2: dense max %.3f, overall max %.3f'
                  % (c, space, np.nanmax(surf[:, 0]), np.nanmax(surf)), flush=True)
    save('G2a_kns_cv_surface', pd.DataFrame(rows).set_index(['country', 'space', 'kappa', 'nonzero']))

    rows_b, rows_c, prs = [], [], []
    for c, (R, m, Z) in samples.items():
        mk = m.to_numpy().ravel()
        Zv = Z.to_numpy()
        p = Zv.shape[1]
        t0 = time.time()
        fits = {}
        for i, s0 in enumerate(starts[c]):
            a_ = slice(s0, s0 + sel.WINDOW)
            Fa, beta, sc = kns.prepare(Zv[a_], mk[a_])
            f_raw = kns.fit_cv(Fa, KAPPAS_ROLL, rotate_pc=False)[0]
            f_pc = kns.fit_cv(Fa, KAPPAS_ROLL, rotate_pc=True)[0]
            fits[i] = (beta, sc, f_raw, f_pc)
        print('  %s %d windows fitted in %.0fs' % (c, len(fits), time.time() - t0), flush=True)
        for sep in SEPS:
            acc = {}
            legs = {}
            for i, a_, b_ in pairs(starts[c], sep):
                beta, sc, f_raw, f_pc = fits[i]
                Fa = kns.prepare(Zv[a_], mk[a_], beta, sc)[0]
                Fb = kns.prepare(Zv[b_], mk[b_], beta, sc)[0]
                mu2, S2 = kns.moments(Fb)
                Rb = R.iloc[b_]
                for space, fit in (('raw', f_raw), ('PC', f_pc)):
                    for col, (b, kap, nz) in fit.items():
                        acc.setdefault((space, col), []).append(kns.r2_oos(mu2, S2, b))
                        if col in ('dense', 8):
                            legs.setdefault('KNS %s %s' % (space, col), []).append(
                                rolling.csr2_F(Rb, np.column_stack([mk[b_], Fb @ b])))
                # random benchmark: k random candidates with an L2 penalty chosen
                # by cross-validation, scored on all candidates (b is zero
                # outside the k chosen)
                for k in kns.SPARSITY:
                    vals = []
                    for _ in range(N_RANDOM_KNS):
                        ix = rng.choice(p, k, replace=False)
                        fr = kns.fit_cv(Fa[:, ix], KAPPAS_ROLL, ks=())[0]['dense']
                        bf = np.zeros(p)
                        bf[ix] = fr[0]
                        vals.append(kns.r2_oos(mu2, S2, bf))
                    acc.setdefault(('random named', k), []).append(np.mean(vals))
                legs.setdefault('random named 1', []).append(np.mean(
                    [rolling.csr2(Rb, mk[b_], Z.iloc[b_], rng.choice(list(Z.columns), 1))
                     for _ in range(db.N_RANDOM)]))
                legs.setdefault('random dense 1', []).append(np.mean(
                    [rolling.csr2_F(Rb, np.column_stack(
                        [mk[b_], Fb @ db.random_directions(p, 1, rng)]))
                     for _ in range(db.N_RANDOM)]))
                prs.append(dict(country=c, separation=sep, window=i,
                                **{'%s %s' % k_: v[-1] for k_, v in acc.items()}))
            shared = max(0, sel.WINDOW - sep * rolling.STEP)
            for (space, col), v in acc.items():
                rows_b.append(dict(country=c, separation=sep, months_shared=shared,
                                   space=space, nonzero=col, R2_oos=np.mean(v),
                                   R2_oos_median=np.median(v), n_pairs=len(v)))
            dense = np.array(acc[('raw', 'dense')])
            for (space, col), v in acc.items():
                # the dense L2 solution is rotation invariant, so PC dense equals raw dense
                if (space, col) not in (('raw', 'dense'), ('PC', 'dense')):
                    rows_c.append(edge_row(country=c, separation=sep, months_shared=shared,
                                           comparison='dense vs %s %s' % (space, col),
                                           metric='KNS R2 eq.30', d=dense - np.array(v)))
            for name in ('KNS raw dense', 'KNS raw 8', 'KNS PC 8'):
                if name in legs:
                    for ref in ('random named 1', 'random dense 1'):
                        rows_c.append(edge_row(country=c, separation=sep, months_shared=shared,
                                               comparison='%s vs %s' % (name, ref),
                                               metric='legs CS R2 (P11)',
                                               d=np.array(legs[name]) - np.array(legs[ref])))
            print('  %s s=%d  R2_oos dense %.3f | raw k=1 %.3f k=8 %.3f | PC k=1 %.3f k=8 %.3f'
                  ' | random k=8 %.3f'
                  % (c, sep, np.mean(acc[('raw', 'dense')]),
                     np.mean(acc.get(('raw', 1), [np.nan])), np.mean(acc.get(('raw', 8), [np.nan])),
                     np.mean(acc.get(('PC', 1), [np.nan])), np.mean(acc.get(('PC', 8), [np.nan])),
                     np.mean(acc[('random named', 8)])), flush=True)
    save('G2b_kns_oos', pd.DataFrame(rows_b).set_index(['country', 'separation', 'space', 'nonzero']))
    save('G2c_kns_edges', pd.DataFrame(rows_c).set_index(['country', 'separation', 'comparison']))
    save('G2_pairs', pd.DataFrame(prs).set_index(['country', 'separation', 'window']))


# ------------------------------------------------------------ lucky factors ---
def lucky(samples, starts):
    print('\nG3 lucky factors', flush=True)
    rows = []
    for c, (R, m, Z) in samples.items():
        for stat in ('median', 'mean'):
            t0 = time.time()
            chosen, steps = lf.select(R.to_numpy(), m.to_numpy().ravel(), Z.to_numpy(),
                                      names=Z.columns, stat=stat, n_boot=lf.N_BOOT,
                                      rng=np.random.default_rng(sel.SEED))
            for s_ in steps:
                rows.append(dict(country=c, statistic=stat, **s_))
            print('  %s SI_%s: %s  (%.0fs)' % (c, stat, chosen or 'nothing', time.time() - t0),
                  flush=True)
    save('G3a_lucky_full_sample', pd.DataFrame(rows).set_index(['country', 'statistic', 'step']))

    sets = []
    for c, (R, m, Z) in samples.items():
        mk = m.to_numpy().ravel()
        cols = list(Z.columns)
        t0 = time.time()
        for i, s0 in enumerate(starts[c]):
            sl = slice(s0, s0 + sel.WINDOW)
            pick, _ = lf.select(R.iloc[sl].to_numpy(), mk[sl], Z.iloc[sl].to_numpy(),
                                names=cols, n_boot=N_BOOT_ROLLING,
                                rng=np.random.default_rng(sel.SEED + i))
            sets.append(dict(country=c, window=i, K=len(pick), members=', '.join(pick)))
        print('  %s rolling windows selected in %.0fs' % (c, time.time() - t0), flush=True)
    save('G3b_lucky_window_sets', pd.DataFrame(sets).set_index(['country', 'window']))


def main():
    rng = np.random.default_rng(sel.SEED)
    scope_coverage()
    U, samples, starts = load(config.COUNTRIES)
    dense_benchmark(samples, starts, rng)
    sdf_shrinkage(samples, starts, rng)
    lucky(samples, starts)


if __name__ == '__main__':
    main()
