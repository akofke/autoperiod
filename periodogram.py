import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import math
from astropy.stats import LombScargle


def main():
    a = np.array([0, 0, 0, 3] * 5, np.float)
    t = np.arange(1, 21, dtype=np.float)

    freq, pwr = LombScargle(t, a).autopower()

    plt.plot(freq, pwr)

    plt.show()


if __name__ == '__main__':
    main()
