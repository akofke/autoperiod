import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import LombScargle
from datetime import datetime
from scipy.signal import fftconvolve, lombscargle
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

    freq = np.linspace(0.0001, 0.01, 10000)
    freq, pwr = LombScargle(times, values).autopower()
    # freq = np.linspace(0.001, 0.5, 100000)
    # pwr = lombscargle(days.astype(float), values, freq)

    # autocorr = fftconvolve(values, values[::-1], mode='full')
    autocorr = np.correlate(values, values, mode='full')
    # autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[autocorr.size/2:]

    plt.subplot(311)
    plt.plot(1/freq, pwr)

    plt.subplot(312)
    plt.plot(times, autocorr)

    plt.subplot(313)
    plt.plot(times, values)

    plt.show()

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
