# TODO
# ask PI if hes fine with forcefully going under the expected length, or maybe use random periods with shorter length by default?
# check for total applied temperature via integral of the stimulus? - how much pain in a given trial? or should it vary?

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def cosine_half_cycle(period, amplitude, y_intercept=0, t_start=0, sample_rate=10):
    frequency = 1 / period
    num_steps = period * sample_rate
    assert num_steps == int(num_steps), "Number of steps must be an integer"
    num_steps = int(num_steps)
    t = np.linspace(0, period, num_steps)
    y = amplitude * np.cos(np.pi * frequency * t) + y_intercept - amplitude
    t += t_start
    return t, y


class StimulusGenerator:
    """
    Generates a stimulus based on appending half cycle cosine functions (thus the term period refers to 1 pi in this class).
    """

    def __init__(self, config=None, seed=None, debug=False):
        self.debug = debug
        if config is None:
            config = {}

        # Initialize parameters
        self.seed = seed if seed is not None else np.random.randint(0, 1000)
        self.rng_numpy = np.random.default_rng(self.seed)

        self.half_cycle_num = config.get("half_cycle_num", 10)
        self.sample_rate = config.get("sample_rate", 10)
        self.period_range = config.get("period_range", [1, 5])
        self.amplitude_range = config.get("amplitude_range", [0.1, 0.9])
        self.median_range = config.get("median_range", 0.3)
        self.big_decreasing_half_cycle_num = config.get("big_decreasing_half_cycle_num", 3)
        self.big_decreasing_half_cycle_period = config.get("big_decreasing_half_cycle_period", 5)
        self.big_decreasing_half_cycle_amplitude = config.get(
            "big_decreasing_half_cycle_amplitude", 0.85
        )
        self.plateau_num = config.get("plateau_num", 3)
        self.plateau_duration = config.get("plateau_duration", 20)

        self.temperature_baseline = config.get("temperature_baseline", 40)
        self.temperature_range = config.get("temperature_range", 3)

        # Calculate expected length of the stimulus
        # NOTE we can make the wave more versatile by forcing the length down (shorten) from the expected length - sampling from under the expected value
        self.expected_length_random_half_cycles = self._get_expected_length_random_half_cycles(
            shorten=15
        )
        self.expected_length_big_decreasing_half_cycles = (
            self._get_expected_length_big_decreasing_half_cycles()
        )
        self.expected_length = self._get_expected_length()

        # Determine big decreasing half cycle indexes
        self.big_decreasing_half_cycle_idx = self._get_big_decreasing_half_cycle_idx()
        self.big_decreasing_half_cycle_idx_for_insert = (
            self._get_big_decreasing_half_cycle_idx_for_insert()
        )

        # Get periods and amplidtudes for the random half cycles with the expected length
        self.periods = self._get_periods()
        self.amplitudes = self._get_amplitudes()

        # Stimulus
        self.y = None  # Placeholder for the stimulus
        self.generate_stimulus()
        if not debug:
            self.add_calibration()
            self.add_plateaus()

    @property
    def duration(self):  # in seconds
        return len(self.y) / self.sample_rate

    @property
    def t(self):
        return np.linspace(0, self.duration, len(self.y))

    @property
    def y_dot(self):
        return np.gradient(self.y, 1 / self.sample_rate)  # dx in seconds

    def _get_expected_length_random_half_cycles(self, shorten):
        return (
            ((self.period_range[0] + (self.period_range[1] - 1)) / 2)
            * (self.half_cycle_num - self.big_decreasing_half_cycle_num)
            * self.sample_rate
        ) - (shorten * self.sample_rate)

    def _get_expected_length_big_decreasing_half_cycles(self):
        return (
            self.big_decreasing_half_cycle_period
            * self.big_decreasing_half_cycle_num
            * self.sample_rate
        )

    def _get_expected_length(self):
        expected_length = (
            self.expected_length_random_half_cycles
            + self.expected_length_big_decreasing_half_cycles
        )
        expected_length -= expected_length % self.sample_rate  # round to nearest sample
        return expected_length

    def _get_big_decreasing_half_cycle_idx(self):
        return np.sort(
            self.rng_numpy.choice(
                range(1, self.half_cycle_num, 2), self.big_decreasing_half_cycle_num, replace=False
            )
        )

    def _get_big_decreasing_half_cycle_idx_for_insert(self):
        """Indices for np.insert"""
        return [i - idx for idx, i in enumerate(self.big_decreasing_half_cycle_idx)]

    def _get_periods(self):
        """Get periods for the half cycles (vectorized)."""
        # TODO: variance check for rAnDoMnEsS?
        counter = 0
        while True:
            counter += 1
            periods = self.rng_numpy.integers(
                self.period_range[0],
                self.period_range[1],
                self.half_cycle_num - self.big_decreasing_half_cycle_num,
            )
            if np.sum(periods) * self.sample_rate == self.expected_length_random_half_cycles:
                break
        periods = np.insert(
            periods,
            self.big_decreasing_half_cycle_idx_for_insert,
            self.big_decreasing_half_cycle_period,
        )
        if self.debug:
            print(f"Periods: {counter} iterations to converge")
        return periods

    def _get_amplitudes(self):
        """
        Get amplitudes for the half cycles (interatively).

        Note that this code it less readable than the vectorized _get_periods, but for the dependent nature of
        the amplitudes on the y_intercepts, looping is much more efficient and much faster than vectorized operations.
        If one value is invalid we do not need to recompute the entire array, just the current value.
        """
        retry_limit_per_half_cycle = 5
        counter = 0
        while True:
            success = True
            amplitudes = []
            y_intercepts = []
            y_intercept = -1  # starting intercept

            for i in range(self.half_cycle_num):
                retries = retry_limit_per_half_cycle
                valid_amplitude_found = False

                while retries > 0 and not valid_amplitude_found:
                    counter += 1
                    if i in self.big_decreasing_half_cycle_idx:
                        amplitude = self.big_decreasing_half_cycle_amplitude
                    else:
                        amplitude = self.rng_numpy.uniform(
                            self.amplitude_range[0], self.amplitude_range[1]
                        )
                        if i % 2 == 0:
                            amplitude *= -1  # invert amplitude for increasing half cycles
                    next_y_intercept = y_intercept + amplitude * -2

                    if (
                        -1 <= next_y_intercept <= 1
                        and -self.median_range
                        <= (next_y_intercept + y_intercept) / 2
                        <= self.median_range
                    ):
                        valid_amplitude_found = True
                        amplitudes.append(amplitude)
                        y_intercepts.append(y_intercept)
                        y_intercept = next_y_intercept
                    else:
                        retries -= 1

                if not valid_amplitude_found:
                    success = False
                    break
            if success and np.max(y_intercepts) > 0.95:
                break
        if self.debug:
            print(f"Amplitudes: {counter} iterations to converge")
        return amplitudes

    def generate_stimulus(self):
        yi = []
        t_start = 0
        y_intercept = -1

        for i in range(self.half_cycle_num):
            period = self.periods[i]
            amplitude = self.amplitudes[i]
            t, y = cosine_half_cycle(period, amplitude, y_intercept, t_start, self.sample_rate)
            y_intercept = y[-1]
            t_start = t[-1]
            yi.append(y)

        self.y = np.concatenate(yi)

    def add_calibration(self):
        self.y *= self.temperature_range / 2
        self.y += self.temperature_baseline
        return self

    def add_plateaus(self):
        """
        Adds plateaus to the stimulus at random positions, but only when the temperature is rising
        and the temperature is between the 25th and 75th percentile. The distance between the
        plateaus is at least 1.5 times the plateau_duration.
        """
        if self.y is None:
            raise ValueError("Waveform not generated. Please run generate_stimulus() first.")

        # Get indices of values within the 25th and 75th percentile and with a rising temperature
        q25, q75 = np.percentile(self.y, 25), np.percentile(self.y, 75)
        idx_iqr_values = np.where((self.y > q25) & (self.y < q75) & (self.y_dot > 0.07))[0]

        # Find suitable positions for the plateaus
        counter = 0
        while True:
            counter += 1
            if counter == 100:
                raise ValueError(
                    "Unable to add the specified number of plateaus within the given wave.\n"
                    "This issue usually arises when the number and/or duration of plateaus is too high.\n"
                    "relative to the plateau_duration of the wave.\n"
                    "Try again with a different seed or change the parameters of the add_plateaus method."
                )
            idx_plateaus = self.rng_numpy.choice(idx_iqr_values, self.plateau_num, replace=False)
            idx_plateaus = np.sort(idx_plateaus)
            # The distance between the plateaus should be at least 1.5 plateau_duration
            if np.all(np.diff(idx_plateaus) > 1.5 * self.plateau_duration * self.sample_rate):
                break

        y_new = []
        for idx, val in enumerate(self.y):
            y_new.append(val)
            if idx in idx_plateaus:
                y_new.extend(np.full(self.plateau_duration * self.sample_rate, val))
        self.y = np.array(y_new)
        return self


def stimulus_extra(stimulus, s_RoC, display_stats=True):
    """
    For plotly graphing of f(x), f'(x), and labels. Also displays the number and length of cooling segments.

    Parameters
    ----------
    f : array_like
        The function values at each time point.
    stimulus.y_dot : array_like
        The derivative of the function at each time point.
    sample_rate : int
        The sample rate of the data.
    s_RoC : float
        The rate of change threshold (°C/s) for alternative labels.
        For more information about thresholds, also see: http://www.scholarpedia.org/article/Thermal_touch#Thermal_thresholds
    display_stats : bool, optional
        If True, the number and length of cooling segments are displayed (default is True).

    Returns
    -------
    labels : array_like
        A binary array where 0 indicates cooling and 1 indicates heating.
    labels_alt : array_like
        A ternary array where 0 indicates cooling, 1 indicates heating, and 2 indicates a rate of change less than s_RoC.
    fig : plotly.graph_objects.Figure
        The plotly figure.

    Examples
    --------
    >>> _ = stimulus_extra(stimulus, s_RoC=0.2)
    """
    time = np.array(range(len(stimulus.y))) / stimulus.sample_rate
    # 0 for cooling, 1 for heating
    labels = (stimulus.y_dot >= 0).astype(int)
    # alternative: 0 for cooling, 1 for heating, 2 for RoC < s_RoC
    labels_alt = np.where(np.abs(stimulus.y_dot) > s_RoC, labels, 2)

    # Plot functions and labels
    fig = go.Figure()
    fig.update_layout(autosize=True, height=300, width=900, margin=dict(l=20, r=20, t=40, b=20))
    fig.update_xaxes(title_text="Time (s)", tickmode="linear", tick0=0, dtick=10)
    fig.update_yaxes(
        title_text=r"Temperature (°C) \ RoC (°C/s)",
        # range=[
        #     n:=min(stimulus.y) if np.sign(min(stimulus.y)) +1 else min(min(stimulus.y), -1),
        #     max(stimulus.y) if np.sign(n) + 1 else abs(n),
        # ]
    )

    func = [stimulus.y, stimulus.y_dot, labels, labels_alt]
    func_names = "f(x)", "f'(x)", "Label", "Label (alt)"
    colors = "royalblue", "skyblue", "springgreen", "violet"

    for idx, i in enumerate(func):
        visible = "legendonly" if idx != 0 else True  # only show the first function by default
        fig.add_scatter(
            x=time, y=i, name=func_names[idx], line=dict(color=colors[idx]), visible=visible
        )
    fig.show()

    # Calculate the number and length of cooling segments from the alternative labels.
    # segment_change indicates where the label changes,
    # segment_number is the cumulative sum of segment_change
    df = pd.DataFrame({"label": labels_alt})
    df["segment_change"] = df["label"].ne(df["label"].shift())
    df["segment_number"] = df["segment_change"].cumsum()

    # group by segment_number and calculate the size of each group
    segment_sizes = df.groupby("segment_number").size()

    # filter the segments that correspond to the label 0
    label_0_segments = df.loc[df["label"] == 0, "segment_number"]
    label_0_sizes = segment_sizes.loc[label_0_segments.unique()]

    # calculate the number and length of segments in seconds
    if display_stats:
        print(
            f"Cooling segments [s] based on 'Label_alt' with a rate of change threshold of {s_RoC} (°C/s):\n"
        )
        print((label_0_sizes / stimulus.sample_rate).describe().apply("{:,.2f}".format))

    return labels, labels_alt, fig
