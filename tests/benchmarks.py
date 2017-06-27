from __future__ import absolute_import

from autoperiod import autoperiod
from autoperiod.helpers import load_gpfs_csv
from pytest import approx

from tests.test_autoperiod import data


def bench_gpfs_reads_mc(benchmark):
    times, values = load_gpfs_csv(data("ub-hpc-6665127-gpfs-reads.csv"))
    period = benchmark(autoperiod, times, values, threshold_method='mc')
    assert period == approx(9469, rel=0.0001)

def bench_gpfs_read_stat(benchmark):
    times, values = load_gpfs_csv(data("ub-hpc-6665127-gpfs-reads.csv"))
    period = benchmark(autoperiod, times, values, threshold_method='statistical')
    assert period == approx(9469, rel=0.0001)
