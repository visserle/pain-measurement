# work in progress

# TODO
# find best ratio of ampliudes and adapt code to temperature range
# define optimal number of big decreases and small decreases
# maybe add min temp to qualify as a big decrease
# via stimuli.wave[stimuli.loc_maxima[stimuli._check_decreases(3)[0]["big"]]] you can get the temperatures of the big decreases
# add temp_criteria to init?
# update doc strings
# change criteria in add plateaus to absolute values based on the temperature range
# maybe refactor into two classes with wave & sitmuli function

"""Stimuli generation for the thermal pain experiment, plus a extra function for plotting and labeling."""

import math
import random
import numpy as np
import pandas as pd
import scipy.signal


class StimuliFunction():
    """
    The `StimuliFunction` class creates a wave-like pattern that can be used for various stimuli functions. 
    
    Its main attribute, `wave`, is a numpy array that is computed by adding a low-frequency baseline wave and 
    a high-frequency modulation wave (optionally with varying periods) together. The resulting `wave` can be 
    modified using different methods in the class, like adding flat regions (plateaus), adjusting the
    baseline temperature, or generalizing the big decreases in temperature to a cosine wave segment.

    Attributes
    ----------
    seed : int
        a random seed for generating random numbers.
    frequencies : numpy.array
        an array of frequencies for the sinusoidal waves, where [0] is the baseline and [1] the modulation.
    periods : numpy.array
        an array of periods of the sinusoidal waves, calculated as 1/frequency.
    temp_range : float
        the temperature range for the stimuli function, from VAS 0 to VAS 70.
    amplitude_proportion : float
        the proportion of the amplitudes of the baseline and modulation.
    amplitudes : numpy.array
        an array of amplitudes for the sinusoidal waves
    baseline_temp : float
        the baseline temperature.
    minimal_desired_duration : float
        the desired minimal duration for the vanilla stimuli function (without add_ methods)
    minimal_duration : float
        the duration for the vanilla stimuli function (without add_ methods), calculated as a multiple of the period of the modulation.
        always use self.duration or self.wave.shape[0] / self.sample_rate to get the actual duration.
    duration (@property) : float
        the actual, up-to-date duration of the stimuli function (wave)
    sample_rate : int
        the sample rate for the stimuli function.
    modulation_n_periods : float
        the number of periods for the modulation.
    random_periods : bool
        a flag to determine if the periods of the modulation is randomized.
    wave : numpy.array
        the stimuli function, calculated as the sum of the baseline and the modulation.
    wave_dot (@property): numpy.array
        the derivative of the stimuli function with respect to time (dx in seconds).
    loc_maxima (@property): list
        a list of peak temperatures (local maximas) in the stimuli function (i.e. self.wave).
    loc_minima (@property): list
        a list of local minima in the stimuli function.

    Methods
    -------
    add_baseline_temp(baseline_temp):
        Adds a baseline temperature to the stimuli function. Should be around VAS = 35. 
        (handled in a seperate function to be able to reuse the same seed for different baseline temperatures)
    add_prolonged_extrema(time_to_be_added, percentage_of_extrema, prolong_type='loc_min'):
        Adds prolonged extrema (either loc_maxima or loc_minima) to the stimuli function.    
    add_plateaus(plateau_duration, n_plateaus):
        Adds plateaus to the stimuli function. 
    generalize_big_decreases(temp_criteria, duration):
        Generalizes big decreases in the stimuli function.

    Example
    -------
    ```python
    import numpy as np

    minimal_desired_duration = 200 # in seconds
    periods = [67, 20] # [0] is the baseline and [1] the modulation; in seconds
    frequencies = 1./np.array(periods)
    temp_range = 3 # VAS 70 - VAS 0
    sample_rate = 60
    random_periods = True
    baseline_temp = 40 # @ VAS 35

    plateau_duration = 20
    n_plateaus = 3

    duration_big_decreases = 20

    stimuli = StimuliFunction(
        minimal_desired_duration=minimal_desired_duration,
        frequencies=frequencies,
        temp_range=temp_range,
        sample_rate=sample_rate,
        random_periods=random_periods,
        seed=seed
    ).add_baseline_temp(
        baseline_temp=baseline_temp
    ).add_plateaus(
        plateau_duration=plateau_duration, 
        n_plateaus=n_plateaus
    ).generalize_big_decreases(
        duration=duration_big_decreases
    )
    ```
	For more information on the resulting stimuli wave use:
	>>> from stimuli_function import stimuli_extra
	>>> _ = stimuli_extra(stimuli.wave, stimuli.wave_dot, stimuli.sample_rate, s_RoC=0.2, display_stats=True)


    Notes
    -----
    Inspecting the stimuli function is best done with a vanilla wave, i.e. without random periods or the add_ methods.
    The function stimuli_extra can be used to plot the stimuli function with plotly.\n
    _________________________________________________________________\n
    The usual range from pain threshold to pain tolerance is about 4 °C. \n
        
    From Andreas Strube's Neuron paper (2023) with stimuli of 8 s and Capsaicin: \n
    "During experiment 1, pain levels were calibrated to achieve \n
    - VAS10 (M = 38.1°C, SD = 3.5°C, Min = 31.8°C, Max = 44.8°C), \n
    - VAS30 (M = 39°C, SD = 3.5°C, Min = 32.2°C, Max = 45.3°C), \n
    - VAS50 (M = 39.9°C, SD = 3.6°C, Min = 32.5°C, Max = 46.2°C) and \n
    - VAS70 (M = 40.8°C, SD = 3.8°C, Min = 32.8°C, Max = 47.2°C) pain levels \n
    - (for highly effective conditioning, test trials, weakly effective conditioning and VAS70 pain stimulation, respectively).\n
    During experiment 2, pain levels were calibrated to achieve \n
    - VAS10 (M = 38.2°C, SD = 3.1°C, Min = 31.72°C, Max = 44.5°C),\n
    - VAS30 (M = 39°C, SD = 3.1°C, Min = 32.1°C, Max = 45.3°C), \n
    - VAS50 (M = 38.19°C, SD = 3.1°C, Min = 32.5°C, Max = 46.1°C) and \n
    - VAS70 (M = 40.5°C, SD = 3.2°C, Min = 32.8°C, Max = 46.9°C) pain levels \n
    - (for highly effective conditioning, test trials, weakly effective conditioning and pain stimulation, respectively)."
    """

    def __init__(
            self,
            minimal_desired_duration,
            frequencies,
            temp_range,
            sample_rate,
            random_periods=True,
            checked_decreases=False,
            seed=None):
        """
        The constructor for StimuliFunction class.

        Parameters
        ----------
        minimal_desired_duration : float
            The minimal desired duration of the wave.
        frequencies : list
            The frequencies of the sinusoidal waves, where [0] is the baseline and [1] the modulation.
        temp_range : float
            The temperature range for the stimuli function, from VAS 0 to VAS 70.
        sample_rate : int, optional
            The sample rate of the wave.
        random_periods : bool, optional
            If True, the periods of the modulation are randomized (default is True).
        checked_decreases : bool, optional
            If True, the stimuli function is checked for the right number of big decreases (default is False (for testing)).
        seed : int, optional
            The seed for the random number generator instances (default is None, which generates a random seed).
        """
        # Class-internal instances of random number generators
        self.rng = random.Random()
        if seed is None:
            self.seed = self.rng.randint(0, 1000)
        else:
            self.seed = seed
        self.rng.seed(self.seed)
        self.rng_numpy = np.random.default_rng(self.seed)

        # Sinusoidal waves where [0] is the baseline and [1] the modulation which gets added on top
        self.frequencies = np.array(frequencies)
        self.periods = 1/self.frequencies

        # Calculate the amplitudes for the sinusoidal waves based on the given temperature range
        self.temp_range = temp_range
        self._amplitude_proportion = 0.375 # value based on empirical testing, see notebook
        self.amplitudes = self._calculate_amplitudes()
        self.temp_criteria = self.temp_range * 0.8 # critical temperature difference for big decreases

        # Sampling rate and actual duration (without add_ methods)
        self.sample_rate = sample_rate
        self.minimal_desired_duration = minimal_desired_duration
        self._calculate_minimal_duration()

        # if True, the periods of the modulation are randomized
        self.random_periods = random_periods
        self.checked_decreases = checked_decreases

        self._create_wave()
        self._wave = self.wave
        self.baseline_temp = 0
        self._duration = self.duration
        self._wave_dot = self.wave_dot

    @property
    def amplitude_proportion(self):
        """The weights for the amplitudes of the baseline and modulation."""
        return self._amplitude_proportion

    @amplitude_proportion.setter
    def amplitude_proportion(self, value):
        """
        Setter for amplitude_proportion.
        This is needed for the setter to work with the sliders in the GUI of the bokeh app in the notebook.
        """
        if not (0 <= value <= 1):  # Ensure the value is within an acceptable range
            raise ValueError("Amplitude proportion must be between 0 and 1.")
        self._amplitude_proportion = value
        self.amplitudes = self._calculate_amplitudes()  # Recalculate the amplitudes
        self._create_wave()

    def _calculate_amplitudes(self):
        """Method to calculate amplitudes based on amplitude_weights"""
        return np.array(
            [self._amplitude_proportion * self.temp_range / 2,
             (1 - self._amplitude_proportion) * self.temp_range / 2])

    def _create_wave(self):
        # Summing self.baseline and self.modulation create the stimuli function self.wave
        counter = 0
        while True: # make sure we exactly have 3 big decreases
            counter += 1
            if counter > 1000:
                raise ValueError(
                    "Unable to create a stimuli function with exactly 3 big decreases.\n"
                )
            self._create_baseline()
            self._create_modulation()
            self.wave = self.baseline + self.modulation
            self._loc_maxima = self.loc_maxima
            self._loc_minima = self.loc_minima
            if self.checked_decreases:
                if self._check_decreases()[0]["big"].size == 3:
                    break
            else:
                break       

    def _calculate_minimal_duration(self):
        """Calculates the true minimal duration ensuring it's a multiple of the period of the modulation
        and the vanilla wave always ends with a local minimum.

        Note: The "true" minimal duration is a multiple of the period of the modulation where the wave always ends with a ramp on (odd number)."""
        modulation_period = self.periods[1]
        self.modulation_n_periods = math.ceil(self.minimal_desired_duration / modulation_period)
        self.modulation_n_periods += 1 - self.modulation_n_periods % 2  # ensure it's odd
        self.minimal_duration = self.modulation_n_periods * modulation_period

    def _create_baseline(self):
        """Creates the baseline sinusoidal wave"""
        time = np.arange(0, self.minimal_duration, 1/self.sample_rate)
        self.baseline = \
            self.amplitudes[0] * -np.cos(time * 2 * np.pi * self.frequencies[0])

    def _create_modulation(self):
        """
        Creates the modulation sinusoidal wave (with varying frequency if random_periods=True).
        The modulation has to be created period-wise as the frequency varies with every period.
        """
        def _noise_that_sums_to_0(n, factor):
            """
            Returns a noise vector that sums to 0 to be added to the periods of the modulation 
            if self.random_periods is True.
            """
            # create noise for n//2
            noise = self.rng_numpy.uniform(
                -factor * self.periods[1], 
                factor * self.periods[1],
                size = n//2)
            # double the noise with inverted values to exactly sum to 0
            noise = np.concatenate((noise, -noise))
            # add 0 if length of n is odd
            if n % 2:
                noise = np.append(noise, 0)
            self.rng.shuffle(noise)
            return np.round(noise)

        modulation_random_factor = _noise_that_sums_to_0(
            n = self.modulation_n_periods,
            factor = 0.6) if self.random_periods else np.zeros(self.modulation_n_periods)

        # create the modulation period-wise
        self.modulation = []
        for i in range(self.modulation_n_periods):
            period = self.periods[1] + modulation_random_factor[i]
            frequency = 1/period
            num_steps = int(period * self.sample_rate)
            time_ = np.linspace(0, period, num_steps) # do not use arange here, floating point errors
            # wave_ has to be inverted every second period to get a sinosoidal wave
            if i % 2 == 0:
                wave_ = \
                    self.amplitudes[1] * -np.cos(np.pi * frequency * time_)
            else:
                wave_ = \
                    self.amplitudes[1] * -np.cos(np.pi * frequency * time_) * -1
            self.modulation.extend(wave_)
        self.modulation = np.array(self.modulation)

    @property
    def duration(self):
        """ The duration of the wave in seconds."""
        self._duration = self.wave.shape[0] / self.sample_rate 
        return self._duration

    @property
    def wave_dot(self):
        self._wave_dot = np.gradient(self.wave, 1/self.sample_rate) # dx in seconds
        return self._wave_dot

    @property
    def loc_maxima(self):
        self._loc_maxima, _ = scipy.signal.find_peaks(self.wave, prominence=0.5)
        return self._loc_maxima

    @property
    def loc_minima(self):
        self._loc_minima, _ = scipy.signal.find_peaks(-self.wave, prominence=0.5)
        return self._loc_minima
    
    @property
    def wave(self):
        return self._wave

    @wave.setter
    def wave(self, new_wave):
        """Always make sure that the wave has the duration of a full second by padding it with the last value."""
        new_wave = np.array(new_wave)
        wave_length = new_wave.size

        # Calculate the required length to reach the next full second
        if wave_length % self.sample_rate != 0:
            required_length = wave_length + (self.sample_rate - (wave_length % self.sample_rate))
            padding_value = new_wave[-1]
            new_wave = np.pad(new_wave, (0, required_length - wave_length), 'constant', constant_values=(padding_value,))
        self._wave = new_wave
    

    def _check_decreases(self):
        """
        Identifies the locations of big and small decreases in temperature 
        between the local maxima and minima of the wave.
        
        Returns
        -------
        idx_decreases : dict
            A dictionary with keys 'big' and 'small' containing the indices of 
            large and small decreases respectively.
        loc_extrema_temps_diff : np.ndarray
            The differences in temperature between local maxima and minima.
        """
        loc_maxima_temps = self.wave[self.loc_maxima]
        loc_minima_temps = self.wave[self.loc_minima]
        loc_extrema_temps_diff = loc_maxima_temps - loc_minima_temps

        idx_decreases = {}
        idx_decreases["big"] = np.where(((loc_extrema_temps_diff) > self.temp_criteria) == 1)[0]
        idx_decreases["small"] = np.where((loc_extrema_temps_diff > self.temp_criteria) == 0)[0]
        return idx_decreases, loc_extrema_temps_diff
    

    def add_baseline_temp(self, baseline_temp):
        """
        Adds a baseline temperature to the wave. It dhould be around VAS = 35.
        """
        self.baseline_temp = baseline_temp
        self.wave += self.baseline_temp
        return self
    
    def add_prolonged_extrema(self, time_to_be_added, percentage_of_extrema, prolong_type='loc_minima'):
        """
        Adds prolonged extrema (either loc_maxima or loc_minima) to the wave.

        Parameters
        ----------
        time_to_be_added : int
            The time to be added per extremum.
        percentage_of_extrema : float
            The percentage of extrema to be prolonged.
        prolong_type : str
            The type of extremum to prolong ('loc_maxima' or 'loc_minima').
        """

        if prolong_type not in ['loc_maxima', 'loc_minima']:
            raise ValueError("Invalid prolong_type. Choose either 'loc_maxima' or 'loc_minima'.")

        extrema = self.loc_maxima if prolong_type == 'loc_maxima' else self.loc_minima
        extrema_chosen = self.rng_numpy.choice(extrema, int(len(extrema) * percentage_of_extrema), replace=False)

        wave_new = []
        for idx, i in enumerate(self.wave):
            wave_new.append(i)
            if idx in extrema_chosen:
                wave_new.extend(
                    [i] * time_to_be_added * self.sample_rate)

        self.wave = np.array(wave_new)
        return self

    def add_plateaus(self, plateau_duration: int, n_plateaus: int):
        """
        Adds plateaus to the wave at random positions, but only when the temperature is rising 
        and the temperature is between the 25th and 75th percentile. The distance between the 
        plateaus is at least 1.5 times the plateau_duration [s].
        """
        def _generate_plateau(start_value):
            """Generate a plateau with the given start value in °C."""
            return np.full(plateau_duration * self.sample_rate, start_value)

        # get indices where the temperature is rising and between the 25th and 75th percentile
        q25, q75 = np.percentile(self.wave, 25), np.percentile(self.wave, 75)
        idx_iqr_values = np.where((self.wave > q25) & (self.wave < q75) & (self.wave_dot > 0.07))[0]

        # find indices for the random plateaus
        counter = 0
        while True:
            counter += 1
            if counter > 1000:
                raise ValueError(
                    "Unable to add the specified number of plateaus within the given wave.\n"
                    "This issue usually arises when the number and/or duration of plateaus is too high "
                    "relative to the length of the wave.\n"
                    "Try again with a different seed or change the parameters of the add_plateaus method.")
            idx_plateaus = self.rng_numpy.choice(idx_iqr_values, n_plateaus, replace=False)
            idx_plateaus = np.sort(idx_plateaus)
            # the distance between the plateaus should be at least 1.5 plateau_duration
            if np.all(np.diff(idx_plateaus) > 1.5 * plateau_duration * self.sample_rate):
                break
                
        wave_new = []
        for idx, i in enumerate(self.wave):
            wave_new.append(i)
            if idx in idx_plateaus:
                wave_new.extend(_generate_plateau(i))
        self.wave = np.array(wave_new)
        return self
    
    def generalize_big_decreases(self, duration: int):
        """
        Modifies the wave to generalize the sections of big decreases (found via _check_decreases) by 
        replacing them with cosine wave segments of specified duration in seconds.
        """
        duration *= self.sample_rate

        idx_decreases, temp_diffs = self._check_decreases()
        idx_big_decreases = idx_decreases["big"]
        if len(idx_big_decreases) == 0:
            ValueError("No big decreases found. Try a lower temp_criteria.")            

        # Create a new list to hold the modified wave
        wave_new = []        
        # Initialize the starting index for the iteration over the original wave
        idx_original = 0

        for j in idx_big_decreases:
            idx_start = self.loc_maxima[j]
            idx_end = self.loc_minima[j]
            # Append the original wave values before the segment to replace
            wave_new.extend(self.wave[idx_original:idx_start])
            # from these two points we want to span a new wave that is exactly duration s long
            x = np.linspace(0, np.pi, duration)
            y = np.cos(x) * temp_diffs[j]/2 + self.wave[idx_start] - temp_diffs[j]/2  # Compute the corresponding y values and shift them to the correct baseline
            # Insert the new wave segment
            wave_new.extend(y)
            # Update the starting index for the next iteration
            idx_original = idx_end

        # Append the remaining original wave values after the last segment
        wave_new.extend(self.wave[idx_original:])
        # Convert the modified wave list to a numpy array
        self.wave = np.array(wave_new)
        return self
    

def stimuli_extra(f, f_dot, sample_rate, s_RoC, display_stats=True):
    """
    For plotly graphing of f(x), f'(x), and labels. Also displays the number and length of cooling segments.
    
    Parameters
    ----------
    f : array_like
        The function values at each time point.
    f_dot : array_like
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
    >>> _ = stimuli_extra(stimuli.wave, stimuli.wave_dot, stimuli.sample_rate, s_RoC=0.2)
    """
    import plotly.graph_objects as go

    time = np.array(range(len(f))) / sample_rate    
    # 0 for cooling, 1 for heating
    labels = (f_dot >= 0).astype(int)
    # alternative: 0 for cooling, 1 for heating, 2 for RoC < s_RoC
    labels_alt = np.where(
        np.abs(f_dot) > s_RoC,
        labels, 2)

    # Plot functions and labels
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        height=300,
        width=900,
        margin=dict(l=20, r=20, t=20, b=20))
    fig.update_xaxes(
        title_text='Time (s)',
        tickmode='linear',
        tick0=0,
        dtick=10)
    fig.update_yaxes(
        title_text=r'Temperature (°C) \ RoC (°C/s)')

    func = [f, f_dot, labels, labels_alt]
    func_names = "f(x)", "f'(x)", "Label", "Label (alt)"
    colors = "royalblue", "skyblue", "springgreen", "violet"

    for idx, i in enumerate(func):
        visible = "legendonly" if idx != 0 else True # only show the first function by default
        fig.add_scatter(
            x=time, y=i,
            name=func_names[idx],
            line=dict(color=colors[idx]),
            visible=visible
        )
    fig.show()

    # Calculate the number and length of cooling segments from the alternative labels.
    # segment_change indicates where the label changes, 
    # segment_number is the cumulative sum of segment_change
    df = pd.DataFrame({'label': labels_alt})
    df['segment_change'] = df['label'].ne(df['label'].shift())
    df['segment_number'] = df['segment_change'].cumsum()

    # group by segment_number and calculate the size of each group
    segment_sizes = df.groupby('segment_number').size()

    # filter the segments that correspond to the label 0
    label_0_segments = df.loc[df['label'] == 0, 'segment_number']
    label_0_sizes = segment_sizes.loc[label_0_segments.unique()]

    # calculate the number and length of segments in seconds
    if display_stats:
        print(f"Cooling segments [s] based on 'Label_alt' with a rate of change threshold of {s_RoC} (°C/s):\n")
        print((label_0_sizes/sample_rate).describe().apply('{:,.2f}'.format))

    return labels, labels_alt, fig
