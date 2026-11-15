"""Paths, data locations and the analysis scope.

Data location
-------------
All input files are read from a data directory. It defaults to ``data/`` in the
repository root and can be moved with the environment variable ``BA_DATA_DIR``.
The expected file names are listed in ``DATA_PATTERNS`` and in the README. If
the files are stored under different names, copy
``config/data_paths.example.ini`` to ``config/data_paths.ini`` (not tracked by
git) and edit the patterns there.

Scope
-----
The thesis runs the same pipeline for four scopes, selected with the
environment variable ``BA_SCOPE`` (default ``main``):

==============  =====================  ==============  ==========================
scope           markets                candidates      used for
==============  =====================  ==============  ==========================
main            DEU, GBR, FRA          61              Chapters 3 to 5, App. A-C
candidates65    DEU, GBR, FRA          65              Table C6
pooled          CHSE (CHE + SWE)       61              Tables D1, D2
bivariate       DEU, GBR, FRA          61              Appendix C, beta check
==============  =====================  ==============  ==========================

Results are written to ``output/<scope>/``.
"""
import configparser
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(REPO, 'config')
DATA_DIR = os.environ.get('BA_DATA_DIR', os.path.join(REPO, 'data'))

#: File name patterns relative to DATA_DIR. {cty} is the lower-case ISO code,
#: {CTY} the upper-case one, {char} a JKP characteristic, {weight} vw_cap or ew.
DATA_PATTERNS = {
    'jkp_factors': 'jkp/[{cty}]_[all_factors]_[monthly]_[vw_cap].zip',
    'jkp_market': 'jkp/[{cty}]_[mkt]_[monthly]_[vw_cap].zip',
    'jkp_portfolios': 'jkp/portfolios/[{cty}]_[{char}]_[monthly]_[{weight}].zip',
    'french_ff5': 'french/Europe_5_Factors_CSV.zip',
    'french_mom': 'french/Europe_Mom_Factor_CSV.zip',
    'fred': 'fred/{series}.csv',
    'compustat_fx': 'compustat/g_exrt_dly.csv.gz',
}

_local = os.path.join(CONFIG_DIR, 'data_paths.ini')
if os.path.exists(_local):
    _cp = configparser.ConfigParser(interpolation=None)
    _cp.read(_local, encoding='utf-8')
    if _cp.has_section('paths'):
        DATA_PATTERNS.update({k: v for k, v in _cp['paths'].items() if k in DATA_PATTERNS})
    if _cp.has_option('data', 'data_dir') and 'BA_DATA_DIR' not in os.environ:
        DATA_DIR = _cp.get('data', 'data_dir')


def data_path(kind, **fields):
    """Absolute path of one input file."""
    pattern = DATA_PATTERNS[kind]
    if 'cty' in fields:
        fields.setdefault('CTY', fields['cty'].upper())
    rel = pattern.format(**fields)
    return rel if os.path.isabs(rel) else os.path.join(DATA_DIR, rel)


def config(name):
    """Absolute path of a file in config/."""
    return os.path.join(CONFIG_DIR, name)


SCOPES = {
    'main': dict(countries=('DEU', 'GBR', 'FRA'), menu='definitional', betas='joint'),
    'candidates65': dict(countries=('DEU', 'GBR', 'FRA'), menu='full', betas='joint'),
    'pooled': dict(countries=('CHSE',), menu='definitional', betas='joint'),
    'bivariate': dict(countries=('DEU', 'GBR', 'FRA'), menu='definitional', betas='bivariate'),
}

SCOPE = os.environ.get('BA_SCOPE', 'main')
if SCOPE not in SCOPES:
    raise ValueError('BA_SCOPE must be one of %s, got %r' % (sorted(SCOPES), SCOPE))
COUNTRIES = list(SCOPES[SCOPE]['countries'])
#: candidate menu: 'definitional' removes the four twins of Table A3, Panel B
#: (61 candidates); 'full' keeps all 65.
MENU = SCOPES[SCOPE]['menu']
#: candidate betas of the penalized selectors: 'joint' (one regression on the
#: market and all candidates, Section 5.1) or 'bivariate' (market and one
#: candidate at a time, the Appendix C check).
CANDIDATE_BETAS = SCOPES[SCOPE]['betas']

OUTPUT = os.path.join(REPO, 'output', SCOPE)


def out(name=''):
    """Absolute path inside the output directory of the current scope."""
    os.makedirs(OUTPUT, exist_ok=True)
    return os.path.join(OUTPUT, name) if name else OUTPUT
