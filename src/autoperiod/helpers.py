from datetime import datetime
import numpy as np


def convert_to_rates(times, values):
    rates = np.diff(values) / np.diff(times)
    return times[:-1], rates


def interpolate_times(times, values):
    times_interp = np.linspace(np.min(times), np.max(times), times.size)
    values_interp = np.interp(times_interp, times, values)
    return times_interp, values_interp


def load_gpfs_csv(filename):
    times, values = np.genfromtxt(filename, delimiter=' ', unpack=True)
    times, values = convert_to_rates(times, values)
    times, values = interpolate_times(times, values)
    return times, values


def load_google_trends_csv(filename):
    data = np.genfromtxt(filename, delimiter=',', converters={
        0: lambda d: datetime.strptime(d, '%Y-%m-%d')
    })
    dates = data['f0']
    values = data['f1']

    times = np.array(map(lambda d: (d - dates[0]).days, dates))
    return times, values
