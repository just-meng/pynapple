"""Micro-benchmark for pynapple core operations: object construction,
``restrict`` (single- and many-interval), indexing, and ``count``.

Runs on synthetic sorted timestamps of increasing size, and reports the median
wall time per call (numba JIT warmed up beforehand, so steady-state is measured
rather than one-off compilation).

Usage::

    python benchmarks/benchmark_core_ops.py
    python benchmarks/benchmark_core_ops.py --sizes 100000 1000000

These numbers back the trusted-construction and searchsorted-``restrict``
optimizations: reconstruction paths (``restrict``/``get``/``count``/...) skip
redundant revalidation, and ``restrict`` finds interval boundaries with
``searchsorted`` + contiguous copies for realistic interval counts, falling back
to the merge scan when intervals are very numerous.
"""

import argparse
import timeit

import numpy as np

import pynapple as nap


def bench(fn, *, min_time=0.25, repeats=7):
    """Median seconds per call, with an adaptive loop count."""
    n = 1
    while timeit.timeit(fn, number=n) < min_time / 5:
        n *= 5
        if n > 5_000_000:
            break
    return float(np.median([timeit.timeit(fn, number=n) / n for _ in range(repeats)]))


def fmt(sec):
    for unit, scale in (("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)):
        if sec * scale >= 1:
            return f"{sec * scale:8.2f} {unit}"
    return f"{sec * 1e9:8.2f} ns"


def make_data(n, rng):
    t = np.cumsum(rng.exponential(1e-3, size=n)).astype(np.float64)
    return t, rng.random(n), rng.random((n, 16))


def warmup(rng):
    t, d, _ = make_data(1000, rng)
    ep = nap.IntervalSet(t[0], t[-1])
    tsd = nap.Tsd(t=t, d=d, time_support=ep)
    tsd.restrict(ep)
    tsd.count(0.1)
    tsd.get(t[10], t[900])
    tsd[100:200]
    nap.TsGroup({0: nap.Ts(t=t, time_support=ep)}, time_support=ep).count(0.1)


def run(sizes):
    rng = np.random.default_rng(0)
    print("pynapple", nap.__version__, "| numpy", np.__version__)
    print("warming up numba JIT...", flush=True)
    warmup(rng)

    for n in sizes:
        print("\n" + "=" * 60)
        print(f"N = {n:,} timestamps")
        print("=" * 60)
        t, d1, d2 = make_data(n, rng)
        t0, t1 = float(t[0]), float(t[-1])
        ep = nap.IntervalSet(t0, t1)
        tsd = nap.Tsd(t=t, d=d1, time_support=ep)
        tsdframe = nap.TsdFrame(t=t, d=d2, time_support=ep)
        ts = nap.Ts(t=t, time_support=ep)

        a, b = float(t[n // 4]), float(t[3 * n // 4])
        one = nap.IntervalSet(a, b)

        def many(m):
            edges = np.linspace(t0, t1, 2 * m + 1)
            return nap.IntervalSet(start=edges[0:-1:2], end=edges[1::2])

        rows = [
            ("Tsd(...) construct", lambda: nap.Tsd(t=t, d=d1, time_support=ep)),
            ("Tsd.restrict (1 window)", lambda: tsd.restrict(one)),
            ("Tsd.restrict (m=100)", lambda mi=many(100): tsd.restrict(mi)),
            ("Tsd.restrict (m=10000)", lambda mi=many(10000): tsd.restrict(mi)),
            ("TsdFrame.restrict (1 window)", lambda: tsdframe.restrict(one)),
            ("Tsd.get(a, b)", lambda: tsd.get(a, b)),
            ("Tsd[slice]", lambda: tsd[n // 4 : 3 * n // 4]),
            ("Tsd.count(bin)", lambda: tsd.count((t1 - t0) / 1000)),
            ("Ts.count(bin)", lambda: ts.count((t1 - t0) / 1000)),
        ]
        for label, fn in rows:
            print(f"  {label:32}: {fmt(bench(fn))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[100_000, 1_000_000, 10_000_000],
        help="timestamp counts to benchmark",
    )
    run(parser.parse_args().sizes)
