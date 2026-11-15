"""Run every test_* function in tests/test_*.py without a test framework.

    python tests/run_tests.py

The test files also work with pytest (python -m pytest tests) if it is
installed. Tests that need the data are skipped when it is missing.
"""
import glob
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from helpers import Skip      # noqa: E402


def main():
    passed, failed, skipped = 0, [], []
    for path in sorted(glob.glob(os.path.join(HERE, 'test_*.py'))):
        name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for fn_name in [n for n in dir(mod) if n.startswith('test_')]:
            fn = getattr(mod, fn_name)
            t0 = time.time()
            try:
                fn()
            except Skip as e:
                skipped.append('%s.%s (%s)' % (name, fn_name, e))
                print('SKIP  %s.%s  %s' % (name, fn_name, e))
                continue
            except Exception:
                failed.append('%s.%s' % (name, fn_name))
                print('FAIL  %s.%s' % (name, fn_name))
                traceback.print_exc()
                continue
            passed += 1
            print('ok    %s.%s  (%.1f s)' % (name, fn_name, time.time() - t0))
    print('\n%d passed, %d failed, %d skipped' % (passed, len(failed), len(skipped)))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
