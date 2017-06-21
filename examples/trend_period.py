import numpy as np
import matplotlib.pyplot as plt

from autoperiod.autoperiod import autocorrelation, get_period_hints, validate_hint
from autoperiod.helpers import load_google_trends_csv

if __name__ == '__main__':
    # values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3] * 100, np.float)
    # values = values / np.max(values)
    # values = 1 - values
    # times = np.arange(1, values.size + 1, dtype=np.float)

    times, values = load_google_trends_csv("test_data/trends_newyears.csv")
    # times, values = load_gpfs_csv("test_data/ub-hpc-6665127-gpfs-reads.csv")

    autocorr = autocorrelation(values)

    fig, ax = plt.subplots(nrows=3, ncols=1)
    hints, periods = get_period_hints(times, values, axes=ax[1])
    for i, period in hints:
        is_valid, period = validate_hint(i, autocorr, periods, times, axes=ax[2], plot_only_valid=False)
        if is_valid:
            break

    ax[0].plot(times, values)

    phase_shift = times[np.argmax(values)]
    amplitude = np.max(values) / 2
    sinwave = np.cos(2 * np.pi / period * (times - phase_shift)) * amplitude + amplitude
    ax[0].plot(times, sinwave)

    fig.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()