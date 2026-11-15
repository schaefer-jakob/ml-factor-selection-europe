"""Run the full replication.

    python run_all.py                          all four scopes, one after the other
    python run_all.py --jobs 4                 the four scopes in parallel
    python run_all.py --scope main             one scope
    python run_all.py --scope main --from 06   resume a scope at a step

Each scope writes its results and a log file to output/<scope>/. The steps
of a scope run in the order listed below, because later steps read the
output of earlier ones. The test-asset screen (scripts/01_test_assets.py) is
not part of the run: its result is fixed in config/.
"""
import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    'main': ['02_ch4_baseline', '03_ch4_fit', '04_ch4_subperiods', '05_ch4_currency',
             '06_ch5_selection', '07_ch5_pca', '08_ch5_benchmarks', '09_ch5_placebo',
             '10_ch5_sharpe', '11_ch5_structure', '12_ch5_themes', '13_figure1',
             '14_appendix_tables', '15_appendix_k_cap'],
    'candidates65': ['06_ch5_selection', '09_ch5_placebo'],
    'pooled': ['02_ch4_baseline', '05_ch4_currency', '06_ch5_selection', '09_ch5_placebo'],
    'bivariate': ['16_appendix_bivariate_betas', '09_ch5_placebo'],
}


def command(scope, start=None):
    """Shell-free list of (step, argv) for one scope."""
    steps = STEPS[scope]
    if start:
        steps = steps[next(i for i, s in enumerate(steps) if s.startswith(start)):]
    return [(s, [sys.executable, '-u', os.path.join(REPO, 'scripts', s + '.py')]) for s in steps]


def run_scope(scope, start=None):
    """Run the steps of one scope in a child process chain; returns the
    Popen of a driver process so that scopes can run in parallel."""
    out_dir = os.path.join(REPO, 'output', scope)
    os.makedirs(out_dir, exist_ok=True)
    driver = [sys.executable, os.path.abspath(__file__), '--scope', scope, '--serial']
    if start:
        driver += ['--from', start]
    return subprocess.Popen(driver)


def run_serial(scope, start=None):
    env = dict(os.environ, BA_SCOPE=scope)
    out_dir = os.path.join(REPO, 'output', scope)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'run.log')
    t_scope = time.time()
    with open(log_path, 'a', encoding='utf-8') as log:
        for step, argv in command(scope, start):
            t0 = time.time()
            log.write('=== %s  %s\n' % (step, time.strftime('%Y-%m-%d %H:%M:%S')))
            log.flush()
            print('[%s] %s' % (scope, step), flush=True)
            rc = subprocess.call(argv, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=REPO)
            log.write('--- %s finished in %.0f s (exit %d)\n' % (step, time.time() - t0, rc))
            log.flush()
            if rc != 0:
                print('[%s] %s failed, see %s' % (scope, step, log_path), flush=True)
                return rc
        log.write('=== scope %s done in %.0f min\n' % (scope, (time.time() - t_scope) / 60))
    print('[%s] done in %.0f min' % (scope, (time.time() - t_scope) / 60), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scope', action='append', choices=sorted(STEPS),
                    help='scope to run (repeatable; default: all)')
    ap.add_argument('--from', dest='start', help='first step of the scope, e.g. 06')
    ap.add_argument('--jobs', type=int, default=1, help='number of scopes run in parallel')
    ap.add_argument('--serial', action='store_true', help=argparse.SUPPRESS)
    a = ap.parse_args()
    scopes = a.scope or list(STEPS)
    if a.serial or a.jobs <= 1:
        for s in scopes:
            if run_serial(s, a.start) != 0:
                sys.exit(1)
        return
    running, failed = [], False
    pending = list(scopes)
    while pending or running:
        while pending and len(running) < a.jobs:
            running.append(run_scope(pending.pop(0), a.start))
        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            running.remove(p)
            failed |= p.returncode != 0
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
