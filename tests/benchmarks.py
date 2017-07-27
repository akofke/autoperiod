from __future__ import absolute_import

from autoperiod import Autoperiod
from autoperiod.helpers import load_gpfs_csv
from pytest import approx

from tests.test_autoperiod import data

_autoperiod = lambda t, v, m: Autoperiod(t, v, threshold_method=m).period

def bench_gpfs_reads_mc(benchmark):
    times, values = load_gpfs_csv(data("ub-hpc-6665127-gpfs-reads.csv"))
    period = benchmark(_autoperiod, times, values, 'mc')
    assert period == approx(9469, rel=0.01)

def bench_gpfs_read_stat(benchmark):
    times, values = load_gpfs_csv(data("ub-hpc-6665127-gpfs-reads.csv"))
    period = benchmark(_autoperiod, times, values, 'stat')
    assert period == approx(9469, rel=0.01)

