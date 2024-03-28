import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy


def cosine_half_cycle(
    period: float,
    amplitude: float,
    y_intercept: float = 0,
    t_start: float = 0,
    sample_rate: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a half cycle of a cosine function."""
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
    Generates a stimulus by appending half cycle cosine functions

    Note that the term `period` refers to 1 pi in this class.
    """

    def __init__(
        self,
        config: dict = None,
        seed: int = None,
        debug: bool = False,
    ):
        self.debug = debug
        if config is None:
            config = {}

        # Initialize parameters
        self.seed = seed if seed is not None else np.random.randint(0, 1000)
        self.rng_numpy = np.random.default_rng(self.seed)

        self.sample_rate = config.get("sample_rate", 10)
        self.half_cycle_num = config.get("half_cycle_num", 10)
        self.period_range = config.get("period_range", [5, 20])
        self.amplitude_range = config.get("amplitude_range", [0.3, 0.9])
        self.inflection_point_range = config.get("inflection_point_range", [-0.5, 0.3])
        self.shorten_expected_duration = config.get("shorten_expected_duration", 7)

        self.big_decreasing_half_cycle_num = config.get(
            "big_decreasing_half_cycle_num", 3
        )
        self.big_decreasing_half_cycle_period = config.get(
            "big_decreasing_half_cycle_period", 20
        )
        self.big_decreasing_half_cycle_amplitude = config.get(
            "big_decreasing_half_cycle_amplitude", 0.85
        )

        self.plateau_num = config.get("plateau_num", 2)
        self.plateau_duration = config.get("plateau_duration", 15)
        self.plateau_percentile_range = config.get("plateau_percentile_range", [25, 50])

        self.prolonged_minima_num = config.get("prolonged_minima_num", 2)
        self.prolonged_minima_duration = config.get("prolonged_minima_duration", 5)

        # Calibrate temperatures
        self.temperature_baseline = float(config.get("temperature_baseline", 40))
        self.temperature_range = float(config.get("temperature_range", 3))

        # Calculate length of the stimulus
        self.desired_length_random_half_cycles = (
            self._get_desired_length_random_half_cycles(
                shorten=self.shorten_expected_duration
            )
        )
        self.desired_length_big_decreasing_half_cycles = (
            self._get_desired_length_big_decreasing_half_cycles()
        )
        self.desired_length = self._get_desired_length()

        # Determine big decreasing half cycle indexes
        self.big_decreasing_half_cycle_idx = self._get_big_decreasing_half_cycle_idx()
        self.big_decreasing_half_cycle_idx_for_insert = (
            self._get_big_decreasing_half_cycle_idx_for_insert()
        )

        # Get periods and amplitudes for the random half cycles with the expected length
        self.periods = self._get_periods()
        self.amplitudes = self._get_amplitudes()

        # Stimulus
        self.y = None  # Placeholder for the stimulus
        self.extensions = []  # Placeholder for the extensions from the add_methods
        self._generate_stimulus()
        if not debug:
            self.add_plateaus()
            self.add_calibration()
            self.add_prolonged_minima()

    @property
    def duration(self):  # in seconds
        return len(self.y) / self.sample_rate

    @property
    def t(self):
        return np.linspace(0, self.duration, len(self.y))

    @property
    def y_dot(self):
        return np.gradient(self.y, 1 / self.sample_rate)  # dx in seconds

    @property
    def big_decreasing_intervals(self) -> list[tuple[int, int]]:
        intervals = []
        for idx in self.big_decreasing_half_cycle_idx:
            start = sum(self.periods[:idx]) * self.sample_rate
            # Adjust the start position based on the extensions that occurred before this index
            start += sum(ext for i, ext in self.extensions if i < start)
            end = start + self.big_decreasing_half_cycle_period * self.sample_rate
            intervals.append((start, end))
        return intervals

    def _get_desired_length_random_half_cycles(
        self,
        shorten,
    ) -> int:
        """
        Get the desired length for the random half cycles.

        Note that shorten [s] is used to force the length down by sampling from
        under the expected value in the _get_periods method.
        """
        desired_length_random_half_cycles = (
            ((self.period_range[0] + (self.period_range[1])) / 2)
            * (self.half_cycle_num - self.big_decreasing_half_cycle_num)
            * self.sample_rate
        ) - (shorten * self.sample_rate)
        desired_length_random_half_cycles -= (
            desired_length_random_half_cycles % self.sample_rate
        )  # necessary for even boundaries, round to nearest sample
        return desired_length_random_half_cycles

    def _get_desired_length_big_decreasing_half_cycles(self) -> int:
        return (
            self.big_decreasing_half_cycle_period
            * self.big_decreasing_half_cycle_num
            * self.sample_rate
        )

    def _get_desired_length(self) -> int:
        desired_length = (
            self.desired_length_random_half_cycles
            + self.desired_length_big_decreasing_half_cycles
        )
        desired_length -= desired_length % self.sample_rate  # round to nearest sample
        return desired_length

    def _get_big_decreasing_half_cycle_idx(self) -> np.ndarray:
        return np.sort(
            self.rng_numpy.choice(
                range(1, self.half_cycle_num, 2),
                self.big_decreasing_half_cycle_num,
                replace=False,
            )
        )

    def _get_big_decreasing_half_cycle_idx_for_insert(self) -> np.ndarray:
        """Indices for np.insert"""
        return [i - idx for idx, i in enumerate(self.big_decreasing_half_cycle_idx)]

    def _get_periods(self) -> np.ndarray:
        """
        Get periods for the half cycles.

        Constraints:
        - The sum of the periods must equal desired_length.
        """
        # TODO: add variance check for rAnDoMnEsS?
        counter = 0
        while True:
            counter += 1
            periods = self.rng_numpy.integers(
                self.period_range[0],
                self.period_range[1],
                self.half_cycle_num - self.big_decreasing_half_cycle_num,
                endpoint=True,
            )
            if (
                np.sum(periods) * self.sample_rate
                == self.desired_length_random_half_cycles
            ):
                break
        periods = np.insert(
            periods,
            self.big_decreasing_half_cycle_idx_for_insert,
            self.big_decreasing_half_cycle_period,
        )
        if self.debug:
            print(f"Periods: {counter} iterations to converge")
        return periods

    def _get_amplitudes(self) -> np.ndarray:
        """
        Get amplitudes for the half cycles (iteratively).

        Note that this code it less readable than the vectorized _get_periods,
        but for the dependent nature of the amplitudes on the y_intercepts,
        looping is much more efficient and much faster than vectorized operations.
        If one intercept is invalid we do not need to recompute the entire array,
        just the current value.

        Contraints:
        - The resulting function must be within -1 and 1.
        - The maximum y_intercept is greater than 0.95.
        - The inflection point of each cosine segment is within inflection_point_range.
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
                        if not i & 1:
                            # invert amplitude for increasing half cycles
                            amplitude *= -1
                    next_y_intercept = y_intercept + amplitude * -2

                    if (
                        -1 <= next_y_intercept <= 1
                        and self.inflection_point_range[0]
                        <= (next_y_intercept + y_intercept) / 2
                        <= self.inflection_point_range[1]
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

    def _generate_stimulus(self):
        """Generates the stimulus based on the periods and amplitudes."""
        yi = []
        t_start = 0
        y_intercept = -1

        for i in range(self.half_cycle_num):
            period = self.periods[i]
            amplitude = self.amplitudes[i]
            t, y = cosine_half_cycle(
                period, amplitude, y_intercept, t_start, self.sample_rate
            )
            y_intercept = y[-1]
            t_start = t[-1]
            yi.append(y)

        self.y = np.concatenate(yi)

    def add_calibration(self):
        """Calibrates temperature range and baseline using participant data."""
        self.y *= self.temperature_range / 2
        self.y += self.temperature_baseline

    def add_plateaus(self):
        """
        Adds plateaus to the stimulus at random positions.

        For each plateau, the temperature is rising and between the given percentile range.
        The distance between the plateaus is at least 1.5 times the plateau_duration.
        """
        # Get indices of values within the given percentile range and with a rising temperature
        percentile_low = np.percentile(self.y, self.plateau_percentile_range[0])
        percentile_high = np.percentile(self.y, self.plateau_percentile_range[1])
        idx_between_values = np.where(
            (self.y > percentile_low)
            & (self.y < percentile_high)
            & (self.y_dot > 0.05)  # only rising temperatures
        )[0]

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
            idx_plateaus = self.rng_numpy.choice(
                idx_between_values, self.plateau_num, replace=False
            )
            idx_plateaus = np.sort(idx_plateaus)
            # The distance between the plateaus should be at least 1.5 plateau_duration
            if np.all(
                np.diff(idx_plateaus) > 1.5 * self.plateau_duration * self.sample_rate
            ):
                break

        self.y = self._extend_stimulus_at_indices(
            indices=idx_plateaus,
            duration=self.plateau_duration,
        )

    def add_prolonged_minima(self):
        """
        Prologue some of the minima in the stimulus to make it more relexaed, less predictable and slightly longer.

        Otherwise, the stimulus can feel like a non-stop series of ups and downs.
        """
        minima_indices, _ = scipy.signal.find_peaks(-self.y, prominence=0.5)
        minima_values = self.y[minima_indices]
        # Find the indices of the smallest minima values
        # (argsort returns indices that would sort the array, and we take the first `self.prolonged_minima_num` ones)
        smallest_minima_indices = np.argsort(minima_values)[: self.prolonged_minima_num]
        prolonged_minima_indices = minima_indices[smallest_minima_indices]

        self.y = self._extend_stimulus_at_indices(
            indices=prolonged_minima_indices,
            duration=self.prolonged_minima_duration,
        )

    def _extend_stimulus_at_indices(
        self,
        indices,
        duration,
    ) -> np.ndarray:
        """Extend the stimulus at specific indices by repeating their values."""
        y_new = np.array([], dtype=self.y.dtype)
        last_idx = 0
        for idx in sorted(indices):
            repeat_count = int(duration * self.sample_rate)
            # Append everything up to the current index
            y_new = np.concatenate((y_new, self.y[last_idx:idx]))
            # Append the repeated value
            y_new = np.concatenate(
                (y_new, np.full(repeat_count, self.y[idx], dtype=self.y.dtype))
            )
            last_idx = idx
            # Track the extensions
            self.extensions.append((idx, repeat_count))
        # Append any remaining values after the last index
        y_new = np.concatenate((y_new, self.y[last_idx:]))
        return y_new


def stimulus_extra(
    stimulus: StimulusGenerator,
    s_RoC: float,
    display_stats: bool = True,
) -> tuple[np.ndarray, np.ndarray, go.Figure]:
    """
    For plotly graphing of f(x), f'(x), and labels. Also displays the number and length of cooling segments.

    Parameters
    ----------
    stimulus : StimulusGenerator
        The stimulus object.
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
    fig.update_layout(
        autosize=True, height=300, width=900, margin=dict(l=20, r=20, t=40, b=20)
    )
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
        visible = (
            "legendonly" if idx != 0 else True
        )  # only show the first function by default
        fig.add_scatter(
            x=time,
            y=i,
            name=func_names[idx],
            line=dict(color=colors[idx]),
            visible=visible,
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


if __name__ == "__main__":
    stimulus = StimulusGenerator()
    _ = stimulus_extra(stimulus, s_RoC=0.2)
