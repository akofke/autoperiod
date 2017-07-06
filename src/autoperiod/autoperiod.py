# coding=utf-8
from __future__ import division, print_function, absolute_import

import math

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import LombScargle
from scipy.signal import fftconvolve
from scipy.stats import linregress
from six.moves import range


# TODO: extract plotting logic into class
def autoperiod(times, values, plot=False, delay_show=False, verbose_plot=False, filename=None, pdfpages=None,
               title=None,
               threshold_method='mc',
               **fig_kw):
    if times[0] != 0:
        # convert absolute times to time differences from the start timestamp
        times = times - times[0]

    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(20, 20) if pdfpages or filename else None,
                                            **fig_kw)
        if title:
            ax1.set_title(title)

    hints, periods = get_period_hints(times, values, threshold_method, axes=ax2 if plot else None)
    acf = autocorrelation(values)

    period = None
    is_valid = False
    for i, p in hints:
        is_valid, period = validate_hint(i, acf, periods, times, axes=ax3 if plot else None,
                                         plot_only_valid=not verbose_plot)
        if is_valid:
            break

    if plot:
        ax1.plot(times, values)
        ax3.plot(times, acf, '-o', lw=0.5, ms=2, label='Autocorrelation')
        ax3.legend()
        if period and is_valid:
            phase_shift = times[np.argmax(values)]
            amplitude = np.max(values) / 2
            sinwave = np.cos(2 * np.pi / period * (times - phase_shift)) * amplitude + amplitude
            ax1.plot(times, sinwave)

        fig.tight_layout()
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        if not delay_show:
            plt.show()

        if pdfpages:
            pdfpages.savefig(fig, dpi=1200, facecolor=fig.get_facecolor())
            plt.close(fig)

        if filename:
            fig.savefig(filename, dpi=1200, format='pdf', facecolor=fig.get_facecolor())
            plt.close(fig)

    return period if is_valid else None


def get_period_hints(times, values, threshold_method='mc', axes=None):
    np.seterr(invalid="ignore")

    permutations = 40
    max_powers = []
    period_hints = []

    time_span = times[-1] - times[0]
    time_interval = times[1] - times[0]

    power_threshold = None
    norm = 1 / (2 * np.var(values - np.mean(values)))

    if threshold_method == 'mc':
        shuf = np.copy(values)
        # TODO: more efficient algorithm for finding the power threshold
        for _ in range(permutations):
            np.random.shuffle(shuf)
            freq, power = LombScargle(times, shuf).autopower(normalization='psd')
            max_powers.append(np.max(power) * norm)

        max_powers.sort()
        power_threshold = max_powers[int(len(max_powers) * .99)]

    freq, power = LombScargle(times, values).autopower(minimum_frequency=1 / time_span,
                                                       maximum_frequency=1 / (time_interval * 2),
                                                       normalization='psd')

    # double the power, since the astropy lomb-scargle implementation halves it during the psd normalization
    power = 2 * power * norm

    if threshold_method == 'statistical':
        power_threshold = -1 * math.log(1 - math.pow(.90, 1 / power.size))

    periods = 1 / freq
    for i, period in enumerate(periods):
        if power[i] > power_threshold and period < time_span / 2 and period > 2 * time_interval:
            period_hints.append((i, period))

    period_hints = sorted(period_hints, key=lambda per: power[per[0]], reverse=True)

    if axes:
        axes.plot(periods, power)
        axes.axhline(power_threshold, color='green', linewidth=1, linestyle='dashed')
        axes.axvline(time_span / 2, c='purple', linewidth=1, linestyle='dashed')
        axes.scatter([p for i, p in period_hints], [power[i] for i, p in period_hints], c='red', marker='x')

    return period_hints, periods


def validate_hint(period_idx, acf, periods, times, axes=None, plot_all_iterations=False, plot_only_valid=True):
    acf /= np.max(acf)
    acf *= 100

    search_min, search_max = get_acf_search_range(period_idx, periods, times)

    min_err = float("inf")
    t_split = None
    min_slope1 = 0
    min_slope2 = 0
    for t in range(search_min + 1, search_max):
        seg1_x = times[search_min:t + 1]
        seg1_y = acf[search_min:t + 1]
        seg2_x = times[t:search_max + 1]
        seg2_y = acf[t:search_max + 1]

        slope1, c1, _, _, stderr1 = linregress(seg1_x, seg1_y)
        slope2, c2, _, _, stderr2 = linregress(seg2_x, seg2_y)

        if stderr1 + stderr2 < min_err and seg1_x.size > 2 and seg2_x.size > 2:
            min_err = stderr1 + stderr2
            t_split = t
            min_slope1 = slope1
            min_slope2 = slope2
            min_c1 = c1
            min_c2 = c2
            min_stderr1 = stderr1
            min_stderr2 = stderr2

            # TODO: if needed for debugging
            # if plot_all_iterations:
            #     plt.figure()
            #     plt.plot(times, acf, label='Autocorrelation')
            #     plt.plot(times[search_min:t+1], c1 + slope1 * times[search_min:t+1], c='r', label='slope: {}, error: {}'.format(slope1, stderr1))
            #     plt.plot(times[t+1:search_max], c2 + slope2 * times[t+1:search_max], c='r', label='slope: {}, error: {}'.format(slope2, stderr2))
            #     plt.scatter(times[t], acf[t], c='g')
            #     plt.legend()

    angle1 = np.arctan(min_slope1) / (np.pi / 2)
    angle2 = np.arctan(min_slope2) / (np.pi / 2)
    valid = min_slope1 > min_slope2 and not np.isclose(np.abs(angle1 - angle2), 0, atol=0.01)
    window = acf[search_min:search_max + 1]
    peak_idx = np.argmax(window) + search_min

    if not plot_only_valid:
        plt.figure()
        plt.plot(times, acf, '-o', lw=.5, ms=2, label='Autocorrelation')
        plt.plot(times[search_min:t_split + 1], min_c1 + min_slope1 * times[search_min:t_split + 1], c='r',
                 label='slope: {}, error: {}'.format(min_slope1, min_stderr1))
        plt.plot(times[t_split:search_max], min_c2 + min_slope2 * times[t_split:search_max], c='r',
                 label='slope: {}, error: {}'.format(min_slope2, min_stderr2))
        plt.scatter(times[t_split], acf[t_split], c='g')
        plt.legend()

    if axes and valid:
        axes.plot(times[search_min:t_split + 1], min_c1 + min_slope1 * times[search_min:t_split + 1], c='r',
                  label='slope: {}, error: {}'.format(min_slope1, min_stderr1))
        axes.plot(times[t_split:search_max], min_c2 + min_slope2 * times[t_split:search_max], c='r',
                  label='slope: {}, error: {}'.format(min_slope2, min_stderr2))
        axes.scatter(times[t_split], acf[t_split], c='g', label='{}'.format(times[t_split]))
        axes.scatter(times[peak_idx], acf[peak_idx], c='y', label='{}'.format(times[peak_idx]))
        axes.legend()

    return valid, times[peak_idx]


def get_acf_search_range(period_index, periods, times):
    min_period = 0.5 * (periods[period_index + 1] + periods[period_index + 2])
    max_period = 0.5 * (periods[period_index - 1] + periods[period_index - 2])

    min_idx = closest_index(min_period, times)
    max_idx = closest_index(max_period, times)
    while max_idx - min_idx + 1 < 6:
        if min_idx > 0:
            min_idx -= 1
        if max_idx < times.size - 1:
            max_idx += 1

    return min_idx, max_idx


def closest_index(value, arr):
    return (np.abs(arr - value)).argmin()


def autocorrelation(values):
    """

    :param values:
    :return:
    """

    acf = fftconvolve(values, values[::-1], mode='full')
    return acf[acf.size // 2:]
