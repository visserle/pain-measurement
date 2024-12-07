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
    "temperature_baseline": 47,
    "temperature_range": 1.5,
}
DEFAULTS.update(DUMMY_VALUES)


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


class StimulusGenerator:
    """
    Generates a stimulus by appending half cycle cosine functions

    Note that the term `period` refers to only 1 pi in this class.
    """

    # NOTE: There is one small detail I missed that would have made the analysis
    # slightly easier: The resulting curve should have been normalized to [-1, 1] for
    # smooth maximum values as determined by the calibration. However, the resulting
    # difference is negligible (for a max value of 47.75 °C, the mean of the resulting max
    # values over all seeds is 47.74 °C). There is no difference in the mean of the min
    # values.

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
        self.rng_numpy = np.random.default_rng(self.seed)

        self.debug = debug

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
        assert self.plateau_duration != self.prolonged_minima_duration, (
            "The plateau duration and prolonged minima duration must be different.",
        )  # bit of a hack to keep the extensions identifiable (important for labeling)

    def _initialize_dynamic_attributes(self):
        """Initializes the dynamic attributes."""
        # Calculate length of the stimulus
        self.desired_length_random_half_cycles = (
            self._get_desired_length_random_half_cycles(
                modify_by=self.shorten_expected_duration
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

    def _get_desired_length_random_half_cycles(
        self,
        modify_by: int,
    ) -> int:
        """
        Get the desired length for the random half cycles.

        Note that modify_by [s] is used to force the length down by sampling from
        under the expected value in the _get_periods method.
        """
        desired_length_random_half_cycles = (
            ((self.period_range[0] + (self.period_range[1])) / 2)
            * (self.half_cycle_num - self.major_decreasing_half_cycle_num)
            * self.sample_rate
        ) - (modify_by * self.sample_rate)
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
        """Indices for np.insert as the array is modified."""
        return [i - idx for idx, i in enumerate(self.major_decreasing_half_cycle_idx)]

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
                            # invert amplitude for the increasing half cycles (cosine)
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
        indices: np.ndarray,
        duration: int,
    ) -> np.ndarray:
        """Extend the stimulus at specific indices by repeating their values."""
        y_new = np.array([], dtype=self.y.dtype)
        last_idx = 0
        # track the number of extensions to adjust the indices accordingly
        extensions_count = 0
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
            self._extensions.append(
                (idx + extensions_count * repeat_count, repeat_count)
            )
            self._extensions.sort()  # sort the extensions by index
            extensions_count += 1
        # Append any remaining values after the last index
        y_new = np.concatenate((y_new, self.y[last_idx:]))
        return y_new

    ####################
    # Labeling section #
    ####################

    @property
    def labels(self) -> dict[str, list[tuple[int, int]]]:
        """Get all the labels for the stimulus in milliseconds."""
        labels = {
            "decreasing_intervals": self.decreasing_intervals_idx,
            "major_decreasing_intervals": self.major_decreasing_intervals_idx,
            "increasing_intervals": self.increasing_intervals_idx,
            "strictly_increasing_intervals": self.strictly_increasing_intervals_idx,
            "strictly_increasing_intervals_without_plateaus": self.strictly_increasing_intervals_without_plateaus_idx,  # noqa E501
            "plateau_intervals": self.plateau_intervals_idx,
            "prolonged_minima_intervals": self.prolonged_minima_intervals_idx,
        }

        def convert_interval(interval: tuple[int, int]) -> tuple[int, int]:
            return tuple(int(t * 1000 / self.sample_rate) for t in interval)

        # Convert indexes to milliseconds
        return {
            key: [convert_interval(interval) for interval in intervals]
            for key, intervals in labels.items()
        }

    @property
    def decreasing_intervals_idx(self) -> list[tuple[int, int]]:
        """
        Get the start and end indices of the decreasing half cycles for labeling.

        This includes the major decreasing half cycles.
        """
        intervals = []
        for idx in range(self.half_cycle_num):
            if self.amplitudes[idx] > 0:
                start = sum(self.periods[:idx]) * self.sample_rate
                for extension in self._extensions:
                    if extension[0] <= start:  # extension before the current half cycle
                        start += extension[1]  # add the extension duration
                end = start + self.periods[idx] * self.sample_rate
                intervals.append((int(start), int(end)))
        return intervals

    @property
    def major_decreasing_intervals_idx(self) -> list[tuple[int, int]]:
        """
        Get the start and end indices of the major decreasing half cycles for labeling.
        """
        # Account for the subset of decreasing periods that are major
        period_indices = np.ceil(self.major_decreasing_half_cycle_idx / 2) - 1
        # Use fancy indexing to get the indices of the major decreasing intervals
        interval_indices = np.array(self.decreasing_intervals_idx)[
            [np.array(period_indices, dtype=int)]
        ]
        # Convert back to list of tuples and ensure all values are regular Python ints
        return [tuple(pair) for pair in interval_indices.reshape(-1, 2)]

    @property
    def increasing_intervals_idx(self) -> list[tuple[int, int]]:
        """
        Get the start and end indices of the increasing half cycles for labeling.
        Accounts for plateaus, prolonged minima, and ensures proper alignment with
        decreasing intervals.

        (This is tricky to implement and relies on other intervals for alignment.)
        """
        intervals = []
        prolonged_minima = self.prolonged_minima_intervals_idx
        plateau_intervals = self.plateau_intervals_idx
        decreasing_intervals = self.decreasing_intervals_idx

        for idx in range(self.half_cycle_num):
            if self.amplitudes[idx] < 0:  # negative because it is a cosine function
                start = sum(self.periods[:idx]) * self.sample_rate

                # Adjust start based on previous intervals and prolonged minima
                if idx > 1:
                    prev_decreasing_end = decreasing_intervals[idx // 2 - 1][1]
                    start = max(start, prev_decreasing_end)

                for minima_start, minima_end in prolonged_minima:
                    if minima_start <= start < minima_end:
                        start = minima_end

                # Calculate end based on the next decreasing interval start
                if idx // 2 < len(decreasing_intervals):
                    end = decreasing_intervals[idx // 2][0]
                else:
                    end = start + self.periods[idx] * self.sample_rate

                # Extend end for plateaus within this increasing interval
                for plateau_start, plateau_end in plateau_intervals:
                    if start <= plateau_start < end:
                        end = max(end, plateau_end)

                intervals.append((int(start), int(end)))

        return intervals

    @property
    def strictly_increasing_intervals_idx(
        self,
    ) -> list[tuple[int, int]]:
        """
        Get the start and end indices of strictly increasing half cycles for labeling.
        """
        intervals = (
            self.strictly_increasing_intervals_without_plateaus_idx
            + self.strictly_increasing_intervals_starting_after_plateaus
        )  # noqa E501

        intervals.sort(key=lambda x: x[0])
        return intervals

    @property
    def strictly_increasing_intervals_without_plateaus_idx(
        self,
    ) -> list[tuple[int, int]]:
        """
        Get the start and end indices of strictly increasing half cycles for labeling.

        A strictly increasing interval is defined as an increasing interval that doesn't
        contain any plateaus. The method filters out increasing intervals that overlap
        with plateau intervals.
        """
        intervals = []
        increasing_intervals = self.increasing_intervals_idx
        plateau_intervals = self.plateau_intervals_idx

        for start, end in increasing_intervals:
            has_plateau = False
            for plateau_start, plateau_end in plateau_intervals:
                # Check if plateau overlaps with the increasing interval
                if not (end <= plateau_start or start >= plateau_end):
                    has_plateau = True

            if not has_plateau:
                intervals.append((int(start), int(end)))

        return intervals

    @property
    def strictly_increasing_intervals_starting_after_plateaus(
        self,
    ) -> list[tuple[int, int]]:
        """
        Get the start and end indices of strictly increasing half cycles for labeling.

        This method is similar to strictly_increasing_intervals_without_plateaus_idx,
        but it only includes strictly increasing intervals that occur after plateaus.
        """
        intervals = []
        increasing_intervals = self.increasing_intervals_idx
        plateau_intervals = self.plateau_intervals_idx

        for start, end in increasing_intervals:
            for plateau_start, plateau_end in plateau_intervals:
                has_plateau = False

                # Check if plateau overlaps with the increasing interval
                if not (end <= plateau_start or start >= plateau_end):
                    has_plateau = True

                    if has_plateau:
                        intervals.append((int(plateau_end), int(end)))

        return intervals

    @property
    def plateau_intervals_idx(self) -> list[tuple[int, int]]:
        """Get the start and end indices of the plateaus for labeling."""
        intervals = []
        prolonged_minima_extension_length = (
            self.prolonged_minima_duration * self.sample_rate
        )
        prolonged_minima_extensions = 0
        for extension in self._extensions:
            # Account for the prolonged minima extensions as they are added last (see
            # self.__init__)
            if extension[1] == prolonged_minima_extension_length:
                prolonged_minima_extensions += 1
            if extension[1] == self.plateau_duration * self.sample_rate:
                intervals.append(  # start and end indices of the plateau
                    (
                        start := extension[0]
                        + (
                            prolonged_minima_extensions
                            * prolonged_minima_extension_length
                        ),
                        start + extension[1],  # length of the plateau
                    )
                )
        return intervals

    @property
    def prolonged_minima_intervals_idx(self) -> list[tuple[int, int]]:
        """Get the start and end indices of the prolonged minima for labeling."""
        # No need to account for other extensions as prolonged minima are added last
        intervals = []
        for extension in self._extensions:
            if extension[1] == self.prolonged_minima_duration * self.sample_rate:
                intervals.append((extension[0], extension[0] + extension[1]))
        return intervals


if __name__ == "__main__":
    # for debugging
    stimulus = StimulusGenerator(seed=396)
    print("Original intervals:", stimulus.increasing_intervals_idx)
