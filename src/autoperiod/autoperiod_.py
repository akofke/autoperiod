# coding=utf-8
from __future__ import division, print_function, absolute_import

import math

import numpy as np
from astropy.stats import LombScargle
from scipy.signal import fftconvolve
from scipy.stats import linregress
from scipy import integrate
from six.moves import range


class Autoperiod(object):

    def __init__(self, times, values, plotter=None):

        # convert absolute times to time differences from start timestamp
        self.times = times - times[0] if times[0] != 0 else times
        self.values = values
        self.plotter = plotter
        self.acf = self.autocorrelation(values)

    def period(self, threshold_method='mc', permutations=40):
        """

        :param threshold_method:
        :return:
        """

    def _get_period_hints(self, method, permutations):
        period_hints = []

        time_span = self.times[-1] - self.times[0]
        time_interval = self.times[1] - self.times[0]

        power_threshold = None
        norm = 1 / (2 * np.var(self.values - np.mean(self.values)))

        freqs, powers = LombScargle(self.times, self.values).autopower(minimum_frequency=1 / time_span,
                                                                       maximum_frequency=1 / (time_interval * 2),
                                                                       normalization='psd')
    def _mc_threshold(self, norm, permutations):
        max_powers = []
        shuf = np.copy(self.values)
        for _ in range(permutations):
            np.random.shuffle(shuf)
            _, powers = LombScargle(self.times, shuf).autopower(normalization='psd')
            max_powers.append(np.max(powers))

        max_powers.sort()
        return max_powers[int(len(max_powers) * .99)] * norm




    @staticmethod
    def autocorrelation(values):
        acf = fftconvolve(values, values[::-1], mode='full')
        return acf[acf.size // 2:]

    @staticmethod
    def closest_index(value, arr):
        return np.abs(arr - value).argmin()
