import numpy as np
import scipy

DEFAULTS = {
    "sample_rate": 10,
    "half_cycle_num": 10,
    "period_range": [5, 20],
    "amplitude_range": [0.3, 1.0],
    "inflection_point_range": [-0.4, 0.3],
    "shorten_expected_duration": 2,
    "major_decreasing_half_cycle_num": 3,
    "major_decreasing_half_cycle_period": 20,
    "major_decreasing_half_cycle_amplitude": 0.925,
    "major_decreasing_half_cycle_min_y_intercept": 0.9,
    "plateau_num": 2,
    "plateau_duration": 15,
    "plateau_percentile_range": [25, 50],
    "prolonged_minima_num": 1,
    "prolonged_minima_duration": 5,
}

DUMMY_VALUES = {
    "temperature_baseline": 40,
    "temperature_range": 3,
}
DEFAULTS.update(DUMMY_VALUES)


class StimulusGenerator:
    """
    Generates a stimulus by appending half cycle cosine functions

    Note that the term `period` refers to only 1 pi in this class.
    """

    def __init__(
        self,
        config: dict | None = None,
        seed: int = None,
        debug: bool = False,
    ):
        # Initialize parameters
        if config is None:
            config = {}
        self.config = {**DEFAULTS, **config}

        self.seed = seed if seed is not None else np.random.randint(100, 1000)
        self.debug = debug
        self.rng_numpy = np.random.default_rng(self.seed)

        # Extract and validate configuration
        for key, value in self.config.items():
            setattr(self, key, value)
        self._validate_parameters()
        self._initialize_dynamic_attributes()

        # Stimulus
        self.y = None  # placeholder for the stimulus
        self._extensions = []  # placeholder for the extensions from the add_methods
        self._generate_stimulus()
        if not debug:
            # add extensions and temperature calibration
            self.add_plateaus()
            self.add_prolonged_minima()
            self.add_calibration()

    def _validate_parameters(self):
        """Validates the configuration parameters."""
        assert (
            self.major_decreasing_half_cycle_min_y_intercept < self.amplitude_range[1]
        ), (
            "The minimum y intercept for the major decreasing half cycles "
            "must be less than the maximum amplitude."
        )
        assert self.major_decreasing_half_cycle_amplitude < self.amplitude_range[1], (
            "The amplitude of the major decreasing half cycles "
            "must be less than the maximum amplitude."
        )

    def _initialize_dynamic_attributes(self):
        """Initializes the dynamic attributes."""
        # Calculate length of the stimulus
        self.desired_length_random_half_cycles = (
            self._get_desired_length_random_half_cycles(
                shorten=self.shorten_expected_duration
            )
        )
        self.desired_length_major_decreasing_half_cycles = (
            self._get_desired_length_major_decreasing_half_cycles()
        )
        self.desired_length = self._get_desired_length()

        # Determine major decreasing half cycle indexes
        self.major_decreasing_half_cycle_idx = (
            self._get_major_decreasing_half_cycle_idx()
        )
        self.major_decreasing_half_cycle_idx_for_insert = (
            self._get_major_decreasing_half_cycle_idx_for_insert()
        )

        # Get periods and amplitudes for the random half cycles with the expected length
        self.periods = self._get_periods()
        self.amplitudes = self._get_amplitudes()

    @property
    def duration(self) -> float:  # in seconds
        return len(self.y) / self.sample_rate

    @property
    def t(self) -> np.ndarray:
        return np.linspace(0, self.duration, len(self.y))

    @property
    def y_dot(self) -> np.ndarray:
        return np.gradient(self.y, 1 / self.sample_rate)  # dx in seconds

    @property
    def major_decreasing_intervals(self) -> list[tuple[int, int]]:
        """
        Get the start and end indices of the major decreasing half cycles for labeling.
        """
        intervals = []
        for idx in self.major_decreasing_half_cycle_idx:
            start = sum(self.periods[:idx]) * self.sample_rate
            for extension in self._extensions:
                if extension[0] <= start:
                    start += extension[1]
            end = start + self.major_decreasing_half_cycle_period * self.sample_rate
            intervals.append((float(start), float(end)))
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
            * (self.half_cycle_num - self.major_decreasing_half_cycle_num)
            * self.sample_rate
        ) - (shorten * self.sample_rate)
        desired_length_random_half_cycles -= (
            desired_length_random_half_cycles % self.sample_rate
        )  # necessary for even boundaries, round to nearest sample
        return desired_length_random_half_cycles

    def _get_desired_length_major_decreasing_half_cycles(self) -> int:
        return (
            self.major_decreasing_half_cycle_period
            * self.major_decreasing_half_cycle_num
            * self.sample_rate
        )

    def _get_desired_length(self) -> int:
        desired_length = (
            self.desired_length_random_half_cycles
            + self.desired_length_major_decreasing_half_cycles
        )
        desired_length -= desired_length % self.sample_rate  # round to nearest sample
        return desired_length

    def _get_major_decreasing_half_cycle_idx(self) -> np.ndarray:
        return np.sort(
            self.rng_numpy.choice(
                range(1, self.half_cycle_num, 2),
                self.major_decreasing_half_cycle_num,
                replace=False,
            )
        )

    def _get_major_decreasing_half_cycle_idx_for_insert(self) -> np.ndarray:
        """Indices for np.insert."""
        return [i - idx for idx, i in enumerate(self.major_decreasing_half_cycle_idx)]

    def _get_periods(self) -> np.ndarray:
        """
        Get periods for the half cycles.

        Constraints:
        - The sum of the periods must equal desired_length.
        """
        # Find periods for the random half cycles by brute force
        counter = 0
        while True:
            counter += 1
            periods = self.rng_numpy.integers(
                self.period_range[0],
                self.period_range[1],
                self.half_cycle_num - self.major_decreasing_half_cycle_num,
                endpoint=True,
            )
            if (
                np.sum(periods) * self.sample_rate
                == self.desired_length_random_half_cycles
            ):
                break
        # Insert the major decreasing half cycle periods
        periods = np.insert(
            periods,
            self.major_decreasing_half_cycle_idx_for_insert,
            self.major_decreasing_half_cycle_period,
        )
        if self.debug:
            print(f"Periods: {counter} iterations to converge")
        return periods

    def _get_amplitudes(self) -> np.ndarray:
        """
        Get amplitudes for the half cycles (iteratively).

        Note that this code it less readable than the vectorized _get_periods,
        but for the dependent nature of the amplitudes on the y_intercepts,
        looping is much more efficient and much faster than vectorized
        brute force operations.
        If one intercept is invalid we do not need to recompute the entire array,
        just the current value.

        Contraints:
        - The resulting function must be within -1 and 1.
        - The y_intercept of each major decrease is greater than
          major_decreasing_half_cycle_min_y_intercept.
        - The inflection point of each cosine segment is within inflection_point_range.
        """
        retry_limit_per_half_cycle = 5
        counter = 0

        while True:
            success = True
            amplitudes = []
            y_intercepts = []
            y_intercept = -1  # starting intercept

            # Iterate over the half cycles
            for i in range(self.half_cycle_num):
                retries = retry_limit_per_half_cycle
                valid_amplitude_found = False

                # Try to find a valid amplitude for the current half cycle
                while retries > 0 and not valid_amplitude_found:
                    counter += 1
                    if i in self.major_decreasing_half_cycle_idx:
                        amplitude = self.major_decreasing_half_cycle_amplitude
                    else:
                        amplitude = self.rng_numpy.uniform(
                            self.amplitude_range[0], self.amplitude_range[1]
                        )
                        if i % 2 == 0:
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

                # If no valid amplitude was found, break the loop and start over
                if not valid_amplitude_found:
                    success = False
                    break
            if not success:
                continue

            # Final check for the major decreasing half cycles
            major_decreases_high_enough = np.all(
                np.array(y_intercepts)[self.major_decreasing_half_cycle_idx]
                > self.major_decreasing_half_cycle_min_y_intercept
            )
            if major_decreases_high_enough:
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
        self.y *= round(self.temperature_range / 2, 2)  # avoid floating point weirdness
        self.y += self.temperature_baseline

    def add_plateaus(self):
        """
        Adds plateaus to the stimulus at random positions.

        For each plateau, the temperature is rising and between the given percentile
        range.
        The distance between the plateaus is at least 1.5 times the plateau_duration.
        """
        # Get indices of values within the given percentile range
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
                    "Unable to add the specified number of plateaus within the given wave.\n"  # noqa E501
                    "This issue usually arises when the number and/or duration of plateaus is too high.\n"  # noqa E501
                    "relative to the plateau_duration of the wave.\n"
                    "Try again with a different seed or change the parameters of the add_plateaus method."  # noqa E501
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
        Prologue some of the minima in the stimulus to make it more relexaed, less
        predictable and slightly longer.

        Otherwise, the stimulus can feel like a non-stop series of ups and downs.
        """
        minima_indices, _ = scipy.signal.find_peaks(-self.y, prominence=0.5)
        minima_values = self.y[minima_indices]
        # Find the indices of the smallest minima values
        # (argsort returns indices that would sort the array,
        # and we take the first `self.prolonged_minima_num` ones)
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
            self._extensions.append((idx, repeat_count))
        # Append any remaining values after the last index
        y_new = np.concatenate((y_new, self.y[last_idx:]))
        return y_new


def cosine_half_cycle(
    period: float,
    amplitude: float,
    y_intercept: float = 0,
    t_start: float = 0,
    sample_rate: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a half cycle cosine function (1 pi) with the given parameters."""
    frequency = 1 / period
    num_steps = period * sample_rate
    assert num_steps == int(num_steps), "Number of steps must be an integer"
    num_steps = int(num_steps)
    t = np.linspace(0, period, num_steps)
    y = amplitude * np.cos(np.pi * frequency * t) + y_intercept - amplitude
    t += t_start
    return t, y


if __name__ == "__main__":
    # for debugging
    stimulus = StimulusGenerator(seed=246)
    print(stimulus.major_decreasing_intervals)
