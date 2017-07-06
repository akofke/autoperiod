import os

import numpy as np
import matplotlib.pyplot as plt

from autoperiod import autoperiod
from autoperiod.helpers import load_google_trends_csv, load_gpfs_csv

if __name__ == '__main__':
    # values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3] * 100, np.float)
    # values = values / np.max(values)
    # values = 1 - values
    # times = np.arange(1, values.size + 1, dtype=np.float)

    times, values = load_google_trends_csv(os.path.join(os.getcwd(), "tests/data/trends_easter.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_google_trends_csv(os.path.join(os.getcwd(), "tests/data/trends_python.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/ub-hpc-6665127-gpfs-writes.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_google_trends_csv(os.path.join(os.getcwd(), "tests/data/trends_newyears.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_google_trends_csv(os.path.join(os.getcwd(), "tests/data/trends_summer.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/industry-2895978-gpfs-reads.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/industry-2901023-gpfs-writes.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/industry-2896041-gpfs-writes.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/chemistry-1455991-gpfs-writes.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/chemistry-1455991-gpfs-writes-cpn-f15-08.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    times, values = load_gpfs_csv(os.path.join(os.getcwd(), "tests/data/ub-hpc-writes-cpn-k16-25-01.csv"))
    p = autoperiod(times, values, plot=True, verbose_plot=False, delay_show=True)

    values = np.array([0, 0, 1, 1] * 10, np.float)
    times = np.arange(0, values.size, dtype=np.float)
    period = autoperiod(times, values, plot=True, delay_show=True, verbose_plot=False, threshold_method='mc')


    # times = np.arange(0, 2, 0.01)
    # values = np.sin(4*np.pi*times)
    plt.show()




