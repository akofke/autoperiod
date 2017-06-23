from __future__ import division

import numpy as np
import os
import pytest
from pytest import approx

from autoperiod import autoperiod
from autoperiod.helpers import load_google_trends_csv

def test_clean_sinwave():
    times = np.arange(0, 10, 0.01)
    values = np.sin(2 * np.pi * times)
    period = autoperiod(times, values)
    assert period == 1.0


@pytest.mark.parametrize(
    "trange,freqscale",
    [
        (20, 2.0),
        (20, 4.0),
        (40, 2.0),
        (40, 4.0)
    ]
)
def test_tight_sinwave(trange, freqscale):
    times = np.arange(0, trange, 0.01)
    values = np.sin(freqscale * np.pi * times)
    period = autoperiod(times, values)
    # TODO: increase precision
    assert period == approx(2.0 / freqscale, abs=2e-2)


def test_squarewave():
    # TODO: this case is very fragile
    values = np.array([0, 0, 1, 1] * 10, np.float)
    times = np.arange(0, values.size, dtype=np.float)
    period = autoperiod(times, values)
    assert period == 4.0


def test_trends_newyears():
    times, values = load_google_trends_csv(os.getcwd() + "/../test_data/trends_newyears.csv")
    period = autoperiod(times, values)
    # within 3% of "expected" period
    assert period == approx(365, rel=0.03)
