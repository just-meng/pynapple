"""Tests that ``restrict`` dispatches to the intended implementation.

``_Base.restrict`` chooses between:
  - ``_restrict_ranges`` (searchsorted boundaries + contiguous copy) for few
    intervals over numpy data, and
  - ``_restrict`` (the numba merge scan) for many intervals or non-numpy data.

These tests spy on the two functions to assert the routing happens, check the
threshold helper, and verify both paths produce identical output.
"""

from unittest.mock import patch

import numpy as np
import pytest

import pynapple as nap
import pynapple.core.base_class as base_class
from pynapple.core._core_functions import _use_searchsorted_restrict

from .helper_tests import MockArray


def _tsd(n=1000):
    t = np.arange(float(n))
    return nap.Tsd(t=t, d=t.copy(), time_support=nap.IntervalSet(0, n - 1))


def _tiled_intervals(n, m):
    """m disjoint intervals tiled over [0, n-1]."""
    edges = np.linspace(0, n - 1, 2 * m + 1)
    return nap.IntervalSet(start=edges[0:-1:2].copy(), end=edges[1::2].copy())


def _spy():
    """Patch both restrict implementations in the base_class namespace, keeping
    their real behavior (wraps=...) so the operation still runs correctly."""
    return (
        patch.object(base_class, "_restrict_ranges", wraps=base_class._restrict_ranges),
        patch.object(base_class, "_restrict", wraps=base_class._restrict),
    )


# --------------------------------------------------------------------------
# threshold helper
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_intervals, n_samples, expected",
    [
        (1, 1000, True),  # 256 < 1000
        (3, 1000, True),  # 768 < 1000
        (4, 1000, False),  # 1024 !< 1000
        (100, 10_000_000, True),
        (100_000, 10_000_000, False),
    ],
)
def test_use_searchsorted_threshold(n_intervals, n_samples, expected):
    assert _use_searchsorted_restrict(n_intervals, n_samples) == expected


# --------------------------------------------------------------------------
# routing (spy)
# --------------------------------------------------------------------------
def test_few_intervals_route_to_searchsorted():
    tsd = _tsd(1000)
    ep = nap.IntervalSet(100, 200)  # 1 interval -> searchsorted path
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        out = tsd.restrict(ep)
    assert ranges.call_count == 1
    assert scan.call_count == 0
    # sanity on the result
    keep = (tsd.t >= 100) & (tsd.t <= 200)
    np.testing.assert_array_equal(out.t, tsd.t[keep])


def test_many_intervals_route_to_scan():
    n = 1000
    tsd = _tsd(n)
    ep = _tiled_intervals(n, 5)  # 5 * 256 = 1280 !< 1000 -> merge scan
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        tsd.restrict(ep)
    assert scan.call_count == 1
    assert ranges.call_count == 0


def test_ts_without_values_routes_to_searchsorted():
    t = np.arange(1000.0)
    ts = nap.Ts(t=t, time_support=nap.IntervalSet(0, 999))
    ep = nap.IntervalSet(100, 200)
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        out = ts.restrict(ep)
    assert ranges.call_count == 1
    assert scan.call_count == 0
    assert len(out) == int(np.sum((t >= 100) & (t <= 200)))


def test_non_numpy_data_routes_to_scan():
    """Array-like (non-ndarray) values must use the scan path even for a single
    interval, since the searchsorted kernel is numpy-only."""
    t = np.arange(1000.0)
    tsd = nap.Tsd(
        t=t,
        d=MockArray(np.arange(1000.0)),
        time_support=nap.IntervalSet(0, 999),
        load_array=False,
    )
    ep = nap.IntervalSet(100, 200)  # few intervals, but non-numpy data
    ranges_patch, scan_patch = _spy()
    with ranges_patch as ranges, scan_patch as scan:
        tsd.restrict(ep)
    assert scan.call_count == 1
    assert ranges.call_count == 0


# --------------------------------------------------------------------------
# both paths must produce identical output (incl. N-D data via jitgather_ranges)
# --------------------------------------------------------------------------
@pytest.fixture(
    params=["ts", "tsd", "tsdframe", "tsdtensor"],
)
def obj(request):
    n = 1000
    t = np.arange(float(n))
    ep = nap.IntervalSet(0, n - 1)
    rng = np.random.default_rng(0)
    if request.param == "ts":
        return nap.Ts(t=t, time_support=ep)
    if request.param == "tsd":
        return nap.Tsd(t=t, d=rng.random(n), time_support=ep)
    if request.param == "tsdframe":
        return nap.TsdFrame(t=t, d=rng.random((n, 4)), time_support=ep)
    return nap.TsdTensor(t=t, d=rng.random((n, 3, 2)), time_support=ep)


@pytest.mark.parametrize(
    "ep",
    [
        nap.IntervalSet(100, 200),  # single window
        nap.IntervalSet(start=[10, 300, 700], end=[50, 400, 900]),  # few windows
    ],
)
def test_searchsorted_and_scan_paths_are_equivalent(obj, ep):
    with patch.object(base_class, "_use_searchsorted_restrict", return_value=True):
        via_searchsorted = obj.restrict(ep)
    with patch.object(base_class, "_use_searchsorted_restrict", return_value=False):
        via_scan = obj.restrict(ep)

    np.testing.assert_array_equal(via_searchsorted.t, via_scan.t)
    if hasattr(obj, "values"):
        np.testing.assert_array_equal(via_searchsorted.d, via_scan.d)
