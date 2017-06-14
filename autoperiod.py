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
    # values = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3] * 100, np.float)
    # times = np.arange(1, 1001, dtype=np.float)

    # data = np.genfromtxt('test_data.csv', delimiter=',', converters={0: read_date}, dtype=[('f0', 'O'), ('f1', '<f8')])

    times, values = np.genfromtxt('ub-hpc-6665127-gpfs-writes.csv', delimiter=' ', unpack=True)
    values = np.diff(values) / np.diff(times)
    times = times[:-1]
    times_interp = np.linspace(np.min(times), np.max(times), times.size)
    values = np.interp(times_interp, times, values)
    times = times_interp


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
    validate_hint(9500, autocorr, times)

    plt.subplot(311)
    plt.plot(freq, pwr)

    plt.subplot(312)
    plt.plot(times, autocorr)

    plt.subplot(313)
    plt.plot(times, values)

    plt.show()

def validate_hint(period, acf, times):
    search_min, search_max = get_acf_search_range(period, times)

    print(search_min)
    print(search_max)
    print("++++")
    for t in range(search_min, search_max):
        slope1, _, _, _, stderr1 = linregress(times[search_min:t+1], acf[search_min:t+1])
        slope2, _, _, _, stderr2 = linregress(times[t+1:search_max], acf[t+1:search_max])
        print(t)
        print("-")
        print(slope1)
        print(stderr1)
        print("--")
        print(slope2)
        print(stderr2)
        print("=====")


def get_acf_search_range(period, periods, times):
    n = times.size
    k = 1 / (period / n)
    min_period = 0.5 * ((n / (k + 1)) + (n / k)) - 1
    max_period = 0.5 * ((n / k) + (n / (k - 1))) + 1
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
