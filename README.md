# Machine Learning Factor Selection in European Equity Markets: replication code

This repository contains the code that produces every number, table and figure
of the bachelor thesis "Machine Learning Factor Selection in European Equity
Markets: Stability and Out-of-Sample Value for Investors" (University of
St. Gallen, 2026). The thesis tests the Fama-French five-factor model with
momentum in Germany, the United Kingdom and France with Fama-MacBeth
regressions on JKP tercile portfolios (Chapter 4). It then selects factors
from 61 JKP candidate factors with the LASSO, the elastic net, forward stepwise
selection and the lucky-factors bootstrap, and measures how stable the
selections are across methods, time and countries and how they price out of
sample (Chapter 5). The data are not included; see [Data](#data).

## Citation

> Schäfer, J. (2026). Machine Learning Factor Selection in European Equity
> Markets: Stability and Out-of-Sample Value for Investors. Bachelor thesis,
> University of St. Gallen.

The data should be cited as Jensen, T. I., Kelly, B., and Pedersen, L. H.
(2023). Is There a Replication Crisis in Finance? The Journal of Finance,
78(5), 2465-2518.

## Requirements

- Python 3.12 (the results were produced with 3.12.10 on Windows 11)
- numpy 2.5.2 and pandas 3.0.5; matplotlib 3.10.3 for Figure 1 only

```
python -m pip install -r requirements.txt
```

The statistical routines (F and chi-squared distributions, coordinate descent
for the LASSO and the elastic net, Newey-West, GRS, Kan-Robotti-Shanken) are
written out in `src/eufactors/`, so there is no dependency on scipy or
scikit-learn.

## Data

| Source | Content | Access |
|---|---|---|
| JKP Global Factor Data, [jkpfactors.com](https://jkpfactors.com) | country factors (all 153) and market factor, monthly, capped value-weighted; tercile portfolios per characteristic, capped value-weighted and equal-weighted | public, downloaded by `scripts/00_download_data.py` |
| Kenneth French Data Library | Europe 5 factors, Europe momentum factor | public, downloaded by `scripts/00_download_data.py` |
| FRED, Federal Reserve Bank of St. Louis | exchange rates 1994 to 1997 (DEXUSUK, DEXSZUS, DEXSDUS, EXGEUS, EXFRUS) and OECD three-month interbank rates (IR3TIB01xxM156N, IRSTCI01CHM156N) | public, downloaded by `scripts/00_download_data.py` |
| Compustat Global, S&P Global Market Intelligence | daily exchange rates 1998 to 2025, all currencies (WRDS: Compustat Global Daily, Exchange Rates; variables `curd`, `datadate`, `exrattpd`, `exratd_tousd`) | WRDS subscription; needed only for the local-currency check (Table B5, Table 3) |

The JKP and French files are publicly available but are not redistributed here.
The thesis uses the JKP vintage with returns through December 2025 (downloaded
July and September 2026). Both providers revise their data, so a later
download can differ slightly from the vintage used.

By default the code expects the files in `data/` below the repository root,
with the original file names:

```
data/
  jkp/[deu]_[all_factors]_[monthly]_[vw_cap].zip      one per market: deu gbr fra che swe
  jkp/[deu]_[mkt]_[monthly]_[vw_cap].zip
  jkp/portfolios/[deu]_[beta_60m]_[monthly]_[vw_cap].zip   one per market and characteristic
  jkp/portfolios/[deu]_[beta_60m]_[monthly]_[ew].zip
  french/Europe_5_Factors_CSV.zip
  french/Europe_Mom_Factor_CSV.zip
  fred/DEXUSUK.csv, ...
  compustat/g_exrt_dly.csv.gz
```

To use another directory, set the environment variable `BA_DATA_DIR`. To use
other file names, copy `config/data_paths.example.ini` to
`config/data_paths.ini` and edit the patterns.

## Replication

```
python scripts/00_download_data.py     # public data; Compustat exchange rates from WRDS by hand
python tests/run_tests.py              # unit tests, about one minute
python run_all.py --jobs 4             # all results, written to output/
```

`run_all.py` runs the scripts of four scopes. A scope fixes the markets, the
candidate set and the candidate betas; results go to `output/<scope>/` together
with a log file `run.log`.

| Scope | Markets | Candidates | Thesis |
|---|---|---|---|
| `main` | DEU, GBR, FRA | 61 | Chapters 3 to 5, Appendices A to C |
| `candidates65` | DEU, GBR, FRA | 65 | Table C6 |
| `pooled` | Switzerland and Sweden pooled | 61 | Appendix D |
| `bivariate` | DEU, GBR, FRA | 61, bivariate candidate betas | Appendix C, beta check |

A single script can be run on its own, for example
`BA_SCOPE=pooled python scripts/06_ch5_selection.py` (PowerShell:
`$env:BA_SCOPE='pooled'; python scripts/06_ch5_selection.py`). Scripts that read
the output of earlier ones say so in their docstring.

`scripts/01_test_assets.py` repeats the test-asset screen of Section 3.2. It is
not part of `run_all.py`, because its result is fixed in
`config/selected_characteristics.csv`; it writes its own copy to `output/main/`
and reports whether the two agree.

All random draws use fixed seeds, so a rerun on the same data reproduces the
results exactly.

## Expected run time

Measured on a Windows machine with 16 logical cores, with the four scopes run in parallel (`--jobs 4`):

| Scope | Run time |
|---|---|
| `main` | 6.2 hours |
| `candidates65` | 5.5 hours |
| `pooled` | 1.8 hours |
| `bivariate` | 3.9 hours |

Chapter 4 takes a few minutes. Most of the time goes into the rolling-window
selections with cross-validated penalties (`06`) and the placebo draws (`09`,
`10`).

## Tables and figures

Output files are in `output/main/` unless the scope is named. Script numbers
refer to `scripts/`.

| Thesis | Content | Script | Output |
|---|---|---|---|
| Table 1 | machine-learning methods in the literature | none | |
| Section 3.2 | 69 and 56 eligible characteristics | 01 | `test_asset_candidates.csv`, `selected_characteristics.csv` |
| Section 3.2 | 375, 382 and 380 complete months | 08 | `G4b_panel_span.csv` |
| Table 2, Panel A | premia, E[f], t(lambda = E[f]) | 02 | `T3_premia_<CTY>.csv` |
| Table 2, Panel A | power against a zero premium | 02 | `T5_power.csv` (`power_diff_sh`) |
| Table 2, Panel B | Wald, GRS, zero-beta rate | 02 | `T4_model_tests.csv` |
| Table 2, Panel B | R^2 with bootstrap interval, share of random models beaten | 03 | `L1_cross_sectional_fit.csv`, `L2_random_factor_hurdle.csv` (k = 6) |
| Table 3, Panel A | design variations with a rejection | 02 | `T7_robustness_grid.csv` |
| Table 3, Panel B | equal weights; KRS; gamma0 = 0; ratio by window | 02 | `T7_robustness_grid.csv`, `T12_krs.csv`, `T11_gamma0_zero.csv`, `premia_ratio_by_window.csv` |
| Table 3, Panel B | Wald p-values in USD, local currency, change of the gaps | 02, 05, 04 | `T4_model_tests.csv`, `FX1_local_currency.csv`, `C1b_lambda_gap_change_wald.csv` |
| Table 4, Panel A | lucky factors | 08 | `G3a_lucky_full_sample.csv` |
| Table 4, Panel A | stable sets, E[V] bound, shared stable factors | 06 | `P12_stable_sets.csv`, `P4_method_axis.csv` |
| Table 4, Panel B | rolling-window set sizes and distinct factors | 06 | `P2_time_axis.csv` |
| Figure 1 | overlap by separation with random-set band | 06, 13 | `P2c_by_separation.csv`, `P2e_overlap_placebo_band.csv`, `figure1_overlap.png` |
| Table 5 | correlation of selection frequencies across markets | 11 | `I3_country_corr_null.csv` |
| Table 6 | out-of-sample R^2 at s = 6 | 06 | `P11_out_of_sample.csv` (`R2_out_selected`) |
| Table 6 | out-of-sample Sharpe ratio at s = 6 | 10 | `S1_oos_sharpe.csv` (tangency) |
| Table 7 | placebo p-values, R^2 | 09 | `P11p_placebo.csv` |
| Table 7 | placebo p-values, Sharpe ratio | 10 | `S1_oos_sharpe.csv` (tangency, s = 6) |
| Section 5.2.1, App. C | selections by theme | 12 | `I5_theme_shares.csv` |
| Section 5.2.3 | mean absolute alpha of the stable LASSO set and the six-factor model | 06 | `P7_vs_benchmark.csv` |
| Table A1 | test-asset characteristics by theme | 14 | `A1_test_assets_by_theme.csv` |
| Table A2 | JKP and French Europe factors | 14 | `A2_jkp_vs_french.csv` |
| Table A3 | candidates by theme | 14 | `A3_candidates_by_theme.csv` |
| Table A4 | descriptions of the candidates | none (JKP documentation) | |
| Table B1 | components of the standard errors | 02 | `B1_standard_errors.csv` |
| Table B2 | robustness grid | 02 | `T7_robustness_grid.csv` |
| Table B3 | power, minimum detectable effects; fit | 02, 03 | `T5_power.csv`, `L1_cross_sectional_fit.csv` |
| Table B4 | gamma0 = 0; full-sample betas with KRS standard errors | 02 | `T11_gamma0_zero.csv`, `T12_krs.csv` |
| Table B5 | US dollar and local currency | 05 | `FX1_local_currency.csv` |
| Table B6 | gaps by sub-period | 04 | `C1_lambda_gap_subperiods.csv`, `C1b_lambda_gap_change_wald.csv` |
| Table B7, Panel A | returns of the beta_60m terciles | 02 | `T1_leg_moments_<CTY>.csv` (`beta_60m_pf1` to `pf3`) |
| Table B7, Panel B | attenuation slope | 02 | `T3_premia_<CTY>.csv` (`attenuation_slope`) |
| Appendix C | cap of eight factors | 15 | `k_cap.csv` |
| Appendix C | rejected correlation threshold of 0.70 | 14 | `C0_candidate_max_corr.csv` |
| Appendix C | no candidate selected by forward stepwise in more than 46 percent of halves | 06 | `P1_stability_frequencies.csv` (column maxima) |
| Appendix C | bivariate candidate betas | 16, 09 | `output/bivariate/P2c_by_separation.csv`, `P11p_placebo.csv` |
| Table C1 | stability-selection frequencies; null band | 06 | `P1_stability_frequencies.csv`, `P5_classics.csv`, `P1b_null_band.csv` |
| Table C2 | steps of the lucky-factors bootstrap | 08 | `G3a_lucky_full_sample.csv` |
| Table C3, Panel A | overlap of clusters | 06 | `P2d_cluster_overlap.csv` |
| Table C3, Panel B | alignment of principal components | 11, 07 | `I2_pc_alignment_null.csv`, `P8b_pca_by_separation.csv` (random subspaces) |
| Table C3, Panel C | exposure structure of the test assets | 11 | `I4a_exposure_rank.csv`, `I4b_exposure_span.csv` |
| Table C4 | Sharpe ratio by weighting scheme and separation | 10 | `S1_oos_sharpe.csv` |
| Table C5 | SDF shrinkage | 08 | `G2a_kns_cv_surface.csv`, `G2b_kns_oos.csv` |
| Table C6 | results with 65 candidates | 06, 09 | `output/candidates65/`: `P12_stable_sets.csv`, `P2c_by_separation.csv`, `P11_out_of_sample.csv`, `P11p_placebo.csv` |
| Table D1 | pooled baseline, common and country-month intercepts | 02, 05 | `output/pooled/`: `T3_premia_CHSE.csv`, `T4_model_tests.csv`, `FX2_pooled_member_intercepts.csv` |
| Table D2 | pooled selection | 06, 09 | `output/pooled/`: `P12_stable_sets.csv`, `P2c_by_separation.csv`, `P11_out_of_sample.csv`, `P11p_placebo.csv` |

Some scripts write a few columns or tables that the thesis does not report,
for example the i.i.d. versions of reported statistics. In
`03_ch4_fit.py` and `08_ch5_benchmarks.py` the unreported parts are kept
because they draw from the same random-number stream as reported ones.

## Repository layout

```
run_all.py                  runs all scripts of all scopes
config/
  selected_characteristics.csv   the 56 test-asset characteristics (Section 3.2)
  test_asset_candidates.csv      the 69 characteristics meeting conditions 1 to 3
  excluded_candidates.csv        the four twins removed from the candidates (Table A3, Panel B)
  jkp_cluster_labels.csv         JKP themes (Jensen et al., 2023)
  data_paths.example.ini         template for local data locations
scripts/                    one script per result set, numbered in run order
src/eufactors/
  config.py                 data locations, scope, output directory
  data.py                   JKP and French readers
  fx.py                     local-currency conversion
  fmb.py                    rolling betas, Fama-MacBeth, Shanken, Newey-West, Wald
  baseline.py               test assets, GRS, the Chapter 4 pipeline
  krs.py                    Kan-Robotti-Shanken standard errors
  power.py                  power and minimum detectable effects
  selection.py              candidates, LASSO, elastic net, forward stepwise, stability selection
  rolling.py                rolling windows, overlap, out-of-sample R^2
  pca.py                    principal components and subspace alignment
  kns.py                    SDF shrinkage of Kozak, Nagel and Santosh (2020)
  lucky_factors.py          bootstrap of Harvey and Liu (2021)
  dense_benchmark.py        random orthonormal directions
  coverage.py               test-asset coverage and candidate counts
tests/                      unit tests (python tests/run_tests.py)
```

`config/jkp_cluster_labels.csv` is the file `Cluster Labels.csv` from the
JKP replication repository (github.com/bkelly-lab/ReplicationCrisis).

## License

The code is released under the MIT License (see `LICENSE`). The data are
subject to the terms of their providers.
