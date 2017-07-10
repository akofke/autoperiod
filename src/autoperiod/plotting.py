import matplotlib.pyplot as plt


class Plotter:

    def __init__(self, title="Autoperiod", filename=None, figsize=(20, 20), verbose=False):
        self.title = title
        self.filename = filename
        self.fig = plt.figure()
        self.figsize = figsize
        self.verbose = verbose

        self.timeseries_ax = plt.subplot2grid((3, 10), (0, 0), colspan=9, xlabel='Times', ylabel='Values')

        self.area_ratio_ax = plt.subplot2grid((3, 10), (0, 9), colspan=1, xticks=(1, 2), xticklabels=("on", "off"))
        self.area_ratio_ax.get_yaxis().set_visible(False)

        self.periodogram_ax = plt.subplot2grid((3, 10), (1, 0), colspan=10, xlabel='Period', ylabel='Power')

        self.acf_ax = plt.subplot2grid((3, 10), (2, 0), colspan=10, xlabel='Lag', ylabel='Correlation')

    def plot_timeseries(self, times, values):
        self.timeseries_ax.plot(times, values, label='Timeseries')
        self.timeseries_ax.legend()

    def plot_sinwave(self, times, sinwave):
        self.timeseries_ax.plot(times, sinwave, label='Estimated Period')
        self.timeseries_ax.legend()

    def plot_area_ratio(self, on_period_area, off_period_area):
        self.area_ratio_ax.bar(1, on_period_area)
        self.area_ratio_ax.bar(2, off_period_area)
        self.area_ratio_ax.legend()

    def plot_periodogram(self, periods, powers, hints, power_threshold, time_threshold):
        self.periodogram_ax.plot(periods, powers, label='Periodogram')
        self.periodogram_ax.scatter([p for i, p in hints], [powers[i] for i, p in hints], c='red', marker='x', label='Period Hints')
        self.periodogram_ax.axhline(power_threshold, color='green', linewidth=1, linestyle='dashed', label='Min Power')
        self.periodogram_ax.axvline(time_threshold, c='purple', linewidth=1, linestyle='dashed', label='Max Period')
        self.periodogram_ax.legend()

    def plot_acf(self, times, acf):
        self.acf_ax.plot(times, acf, '-o', lw=0.5, ms=2, label='Autocorrelation')
        self.acf_ax.legend()

    def plot_acf_validation(self, times, acf, times1, m1, c1, err1, times2, m2, c2, err2, split_idx, peak_idx):
        self.acf_ax.plot(times1, c1 + m1 * times1, c='r', label='Slope: {}, Error: {}'.format(m1, err1))
        self.acf_ax.plot(times2, c2 + m2 * times2, c='r', label='Slope: {}, Error: {}'.format(m2, err2))
        self.acf_ax.scatter(times[split_idx], acf[split_idx], c='y', label='Split point: {}'.format(times[split_idx]))
        self.acf_ax.scatter(times[peak_idx], acf[peak_idx], c='g', label='Peak point: {}'.format(times[peak_idx]))
        self.acf_ax.legend()

    def show(self):

        self.fig.tight_layout()

        if self.filename:
            self.fig.set_size_inches(*self.figsize)
            self.fig.savefig(self.filename, format='pdf', facecolor=self.fig.get_facecolor())
        plt.show()
