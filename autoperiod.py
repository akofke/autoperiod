from __future__ import division, print_function

import numpy as np
import math
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from datetime import datetime
from scipy.signal import fftconvolve, lombscargle
from scipy.stats import linregress
import astropy.units as u

from six.moves import range


def main():
    # values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3] * 100, np.float)
    # values = values / np.max(values)
    # values = 1 - values
    # times = np.arange(1, values.size + 1, dtype=np.float)

    # data = np.genfromtxt('test_data.csv', delimiter=',', converters={0: read_date}, dtype=[('f0', 'O'), ('f1', '<f8')])

    times, values = np.genfromtxt('ub-hpc-6665127-gpfs-writes.csv', delimiter=' ', unpack=True)
    values = np.diff(values) / np.diff(times)
    times = times[:-1]
    times_interp = np.linspace(np.min(times), np.max(times), times.size)
    values = np.interp(times_interp, times, values)
    times = times_interp

    plt.plot(times, values)
    plt.show()

    # times = process_dates(data)
    # values = data['f1']

    freq, pwr = LombScargle(times, values).autopower(minimum_frequency=0.000001, maximum_frequency=0.01)
    # freq = np.linspace(0.001, 0.5, 100000)
    # pwr = lombscargle(days.astype(float), values, freq)

    # autocorr = fftconvolve(values, values[::-1], mode='full')
    autocorr = np.correlate(values, values, mode='full')
    # autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[autocorr.size//2:]
    print(values.size)
    print(autocorr.size)
    periods = 1 / freq
    validate_hint(41, autocorr, periods, times)

    plt.subplot(311)
    plt.plot(freq, pwr)

    plt.subplot(312)
    plt.plot(times, autocorr)

    plt.subplot(313)
    plt.plot(times, values)

    plt.show()

def get_period_hints(times, values):
    permutations = 100
    max_powers = []
    periods = []

    sequence = np.stack((times, values), axis=-1)

    for _ in range(permutations):
        p = np.random.permutation(sequence)
        freq, power = LombScargle(sequence[:,0], sequence[:,1]).autopower()
        max_powers.append(np.max(power))

    max_powers.sort()
    power_threshold = max_powers[int(len(max_powers) * .99)]

    freq, power = LombScargle(times, values).autopower()

    for i, p in enumerate(1 / freq):
        if power[i] > power_threshold:
            periods.append((i, p))



def validate_hint(period_idx, acf, periods, times, show=True):
    search_min, search_max = get_acf_search_range(period_idx, periods, times)

    print(search_min)
    print(search_max)
    print("++++")

    min_err = float("inf")
    t_split = None
    for t in range(search_min + 1, search_max):
        seg1_x = times[search_min:t+1]
        seg1_y = acf[search_min:t+1]
        seg2_x = times[t+1:search_max+1]
        seg2_y = acf[t+1:search_max+1]

        slope1, c1, _, _, stderr1 = linregress(seg1_x, seg1_y)
        slope2, c2, _, _, stderr2 = linregress(seg2_x, seg2_y)

        print("err1: {}".format(stderr1))
        print("err2: {}".format(stderr2))
        print("sum: {}".format(stderr1 + stderr2))
        print("min: {}".format(min_err))

        if stderr1 + stderr2 < min_err and seg1_x.size > 2 and seg2_x.size > 2:
            min_err = stderr1 + stderr2
            t_split = t
            min_slope1 = slope1
            min_slope2 = slope2
            min_c1 = c1
            min_c2 = c2
            print("<<<<<<<<<")
            print(stderr1)
            print(stderr2)
            print(min_err)
            print(seg1_x.size)
            print(seg2_x.size)
            print(">>>>>>>>>>>")

        if show:
            print(t)
            print("-")
            print(slope1)
            print(stderr1)
            print("--")
            print(slope2)
            print(stderr2)
            print("=====")
            plt.plot(times, acf)
            plt.plot(times[search_min:t+1], c1 + slope1 * times[search_min:t+1], 'r')
            plt.plot(times[t+1:search_max], c2 + slope2 * times[t+1:search_max], 'r')
            plt.scatter(times[t], acf[t], c='g')
            plt.show()

    print(t_split)
    print(times[t_split])
    plt.scatter(times, acf, s=2)
    plt.plot(times[search_min:t_split+1], min_c1 + min_slope1 * times[search_min:t_split+1], 'r')
    plt.plot(times[t_split+1:search_max], min_c2 + min_slope2 * times[t_split+1:search_max], 'r')
    plt.scatter(times[t_split], acf[t_split], c='g')
    plt.show()

def get_acf_search_range(period_index, periods, times):
    min_period = 0.5 * (periods[period_index + 1] + periods[period_index + 2])
    max_period = 0.5 * (periods[period_index - 1] + periods[period_index - 2])

    return closest_index(min_period, times), closest_index(max_period, times)





def closest_index(value, arr):
    return (np.abs(arr - value)).argmin()


def interpolate(times, values):
    """

    :param times:
    :param values:
    :return: a tuple of the new interpolated times and interpolated values
    """
    interpolated_times = np.linspace(np.min(times), np.max(times), times.size)
    return interpolated_times, np.interp(interpolated_times, times, values)

def autocorrelation(values):
    """

    :param values:
    :return:
    """

    return fftconvolve(values, values[::-1], mode='full')[values.size / 2:]

def process_dates(csvdata):
    dates = csvdata['f0']
    first_date = dates[0]
    return np.array(map(lambda d: (d - first_date).days, dates))


def read_date(date_string):
    return datetime.strptime(date_string, '%Y-%m-%d')


def unix_time(dt):
    epoch = datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()

if __name__ == '__main__':
    main()
