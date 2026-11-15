"""Shared helpers for the tests."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import pytest
    Skip = pytest.skip.Exception
except ImportError:
    class Skip(Exception):
        """Raised by a test that cannot run, e.g. because the data are missing."""


def requires_data():
    from eufactors import baseline
    probe = baseline.leg_path('DEU', baseline.CHARS[0])
    if not os.path.exists(probe):
        raise Skip('data not found (%s)' % probe)
