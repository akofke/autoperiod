from __future__ import division

import numpy as np
import os
import pytest

from autoperiod import Autoperiod
from pytest import approx

from autoperiod.helpers import load_google_trends_csv, load_gpfs_csv


def data(name):
    return os.path.join(os.path.dirname(__file__), "data", name)

def test_clean_sinwave():
    times = np.arange(0, 10, 0.01)
    values = np.sin(2 * np.pi * times)
    period = Autoperiod(times, values).period
    assert period == approx(1.0, abs=0.1)


@pytest.mark.parametrize(
    "trange,freqscale",
    [
        (20, 2.0),
        (20, 4.0),
        (40, 2.0),
        (40, 4.0)
    ]
)
@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_tight_sinwave(trange, freqscale, threshold_method):
    times = np.arange(0, trange, 0.01)
    values = np.sin(freqscale * np.pi * times)
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    # TODO: increase precision
    assert period == approx(2.0 / freqscale, abs=2e-2)


@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_squarewave(threshold_method):
    # TODO: this case is very fragile
    values = np.array([0, 0, 1, 1] * 10, np.float)
    times = np.arange(0, values.size, dtype=np.float)
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    assert period == 4.0


@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_trends_newyears(threshold_method):
    times, values = load_google_trends_csv(data("trends_newyears.csv"))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    # within 3% of "expected" period
    assert period == approx(365, rel=0.03)


@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_trends_easter(threshold_method):
    times, values = load_google_trends_csv(data("trends_easter.csv"))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    # Easter isn't a fixed holiday, so the "expected" period won't be as close to 365 days
    assert period == approx(365, rel=0.05)


@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_gpfs_reads(threshold_method):
    times, values = load_gpfs_csv(data("ub-hpc-6665127-gpfs-reads.csv"))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    assert period == approx(9469, rel=0.01)


@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_gpfs_writes(threshold_method):
    times, values = load_gpfs_csv(data("ub-hpc-6665127-gpfs-writes.csv"))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    assert period == approx(9560, rel=0.01)


@pytest.mark.parametrize("threshold_method", ["mc", "stat"])
def test_trends_python_nonperiodic(threshold_method):
    times, values = load_google_trends_csv(data("trends_python.csv"))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    assert period is None


@pytest.mark.parametrize("threshold_method", ['mc', 'stat'])
@pytest.mark.parametrize('filename,expect_period', [
    ('industry-2895978-gpfs-reads.csv', 180),
    ('industry-2896041-gpfs-writes.csv', 150)
])
def test_pcp_smallperiod(threshold_method, filename, expect_period):
    # test for regression to false alarm large period
    times, values = load_gpfs_csv(data(filename))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    assert period == approx(expect_period, rel=0.01)

@pytest.mark.parametrize("mthd", ['mc', 'stat'])
def test_pcp_spiky_acf(mthd):
    times, values = load_gpfs_csv(data("chemistry-1455991-gpfs-writes.csv"))
    period = Autoperiod(times, values, threshold_method=mthd).period
    assert period == approx(3660, rel=0.01)

@pytest.mark.parametrize("threshold_method", ['mc', 'stat'])
@pytest.mark.parametrize('filename', [
    'ub-hpc-writes-cpn-k16-25-01.csv',
])
def test_pcp_noperiod(threshold_method, filename):
    times, values = load_gpfs_csv(data(filename))
    period = Autoperiod(times, values, threshold_method=threshold_method).period
    assert period is None
