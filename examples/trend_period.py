import os

import numpy as np
import matplotlib.pyplot as plt

from autoperiod import autoperiod
from autoperiod.helpers import load_google_trends_csv

if __name__ == '__main__':
    # values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3] * 100, np.float)
    # values = values / np.max(values)
    # values = 1 - values
    # times = np.arange(1, values.size + 1, dtype=np.float)

    # values = np.array([0, 0, 1, 1] * 10, np.float)
    # times = np.arange(0, values.size, dtype=np.float)

    times, values = load_google_trends_csv(os.path.join(os.getcwd(), "test_data/trends_newyears.csv"))
    # times, values = load_gpfs_csv("test_data/ub-hpc-6665127-gpfs-reads.csv")

    # times = np.arange(0, 2, 0.01)
    # values = np.sin(4*np.pi*times)

    autoperiod(times, values, plot=True, verbose_plot=False)



