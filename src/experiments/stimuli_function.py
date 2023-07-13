import math
import random
import numpy as np
import pandas as pd
import scipy.signal


class StimuliFunction():
    """
    The `StimuliFunction` class represent stimuli functions with sinusoidal waves and plateaus.


    Attributes
    ----------
    seed : int
        a random seed for generating random numbers.
    frequencies : numpy.array
        an array of frequencies for the sinusoidal waves, where [0] is the baseline and [1] the modulation.
    periods : numpy.array
        an array of periods of the sinusoidal waves, calculated as 1/frequency.
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
    peaks (@property): list
        a list of peak temperatures (local maxima) in the stimuli function (i.e. self.wave).
    troughs (@property): list
        a list of troughs (local minima) in the stimuli function (a trough is the opposite of a peak).

    Methods
    -------
    _create_baseline():
        Creates the baseline sinusoidal wave.
    _create_modulation():
        Creates the modulation sinusoidal wave (with varying periods if random_periods=True).
    wave_dot():
        Calculates the derivative of the stimuli function with respect to time (dx in seconds).
    add_baseline_temp(baseline_temp):
        Adds a baseline temperature to the stimuli function. Should be around VAS = 35. 
        (handled in a seperate function to be able to reuse the same seed for different baseline temperatures)
    add_prolonged_peaks(time_to_be_added_per_peak, percetage_of_peaks):
        Adds prolonged peaks to the stimuli function. Not used for now.
    add_plateaus(plateau_duration, n_plateaus, add_at_start="random", add_at_end=True):
        Adds plateaus to the stimuli function. 

    Example
    -------
    ````python
    import numpy as np
    minimal_desired_duration = 200 # in seconds
    periods = [67, 20] # [0] is the baseline and [1] the modulation; in seconds
    frequencies = 1./np.array(periods)
    amplitudes = [1, 1.5] # temp range is 2 * sum(amplitudes): max @ VAS 70, min @ VAS 0
    sample_rate = 60
    seed = 463 # use None for random seed
    baseline_temp = 39.2 # @ VAS 35

    stimuli = StimuliFunction(
        minimal_desired_duration, 
        frequencies, 
        amplitudes,
        sample_rate,
        random_periods=True, 
        seed=seed)
    stimuli.add_baseline_temp(baseline_temp)
    stimuli.add_plateaus(plateau_duration=20, n_plateaus=4, add_at_start="random", add_at_end=True)
    ````
	For more information on the resulting stimuli wave use:
	>>> from stimuli_function import stimuli_extra
	>>> _ = stimuli_extra(stimuli.wave, stimuli.wave_dot, stimuli.sample_rate, s_RoC=0.2)


    Notes
    -----
    Inspecting the stimuli function is best done with a vanilla wave, i.e. without random periods or the add_ methods.
    The function stimuli_extra can be used to plot the stimuli function with plotly.\n

    The usual range from pain threshold to pain tolerance is about 4 °C. \n
        
    From Andreas Strube's Neuron paper (2023) with stimuli of 8 s and Capsaicin: \n
    "During experiment 1, pain levels were calibrated to achieve \n
        VAS10 (M = 38.1°C, SD = 3.5°C, Min = 31.8°C, Max = 44.8°C), \n
        VAS30 (M = 39°C, SD = 3.5°C, Min = 32.2°C, Max = 45.3°C), \n
        VAS50 (M = 39.9°C, SD = 3.6°C, Min = 32.5°C, Max = 46.2°C) and \n
        VAS70 (M = 40.8°C, SD = 3.8°C, Min = 32.8°C, Max = 47.2°C) pain levels \n
        (for highly effective conditioning, test trials, weakly effective conditioning and VAS70 pain stimulation, respectively).\n
    During experiment 2, pain levels were calibrated to achieve \n
        VAS10 (M = 38.2°C, SD = 3.1°C, Min = 31.72°C, Max = 44.5°C),\n
        VAS30 (M = 39°C, SD = 3.1°C, Min = 32.1°C, Max = 45.3°C), \n
        VAS50 (M = 38.19°C, SD = 3.1°C, Min = 32.5°C, Max = 46.1°C) and \n
        VAS70 (M = 40.5°C, SD = 3.2°C, Min = 32.8°C, Max = 46.9°C) pain levels \n
        (for highly effective conditioning, test trials, weakly effective conditioning and pain stimulation, respectively)."
    """

    def __init__(self, minimal_desired_duration, 
                 frequencies, amplitudes, sample_rate,
                 random_periods=True, seed=None):
        """
        The constructor for StimuliFunction class.

        Parameters
        ----------
        minimal_desired_duration : float
            The minimal desired duration of the wave.
        frequencies : list
            The frequencies of the sinusoidal waves, where [0] is the baseline and [1] the modulation.
        amplitudes : list
            The amplitudes of the sinusoidal waves, where [0] is the baseline and [1] the modulation.
        sample_rate : int, optional
            The sample rate of the wave.
        random_periods : bool, optional
            If True, the periods of the modulation are randomized (default is True).
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
        self.amplitudes = np.array(amplitudes)

        # Duration and sampling (without add_ methods)
        self.minimal_desired_duration = minimal_desired_duration
        # the "true" minimal duration is a multiple of the period of the modulation
        self.minimal_duration = math.ceil(self.minimal_desired_duration / self.periods[1]) * self.periods[1]
        self.sample_rate = sample_rate

        # Additional variables
        self.modulation_n_periods = math.floor(self.minimal_duration / self.periods[1]) # always int in contrast to // operator
        # if True, the periods of the modulation are randomized
        self.random_periods = random_periods

        # Summing self.baseline and self.modulation create the stimuli function self.wave
        self._create_baseline()
        self._create_modulation()
        self.wave = self.baseline + self.modulation
        self._duration = self.duration
        self._wave_dot = self.wave_dot
        self._peaks = self.peaks
        self._troughs = self.troughs
        

    def _create_baseline(self):
        """Creates the baseline sinusoidal wave"""
        time = np.arange(0, self.minimal_duration, 1/self.sample_rate)
        self.baseline = self.amplitudes[0] * np.sin(
            time * 2 * np.pi * self.frequencies[0])

    def _create_modulation(self):
        """
        Creates the modulation sinusoidal wave (with varying frequency if random_periods=True).
        The modulation has to be created period-wise as the frequency varies with every period
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
            if n % 2 == 1:
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
            time_ = np.arange(0, period, 1/self.sample_rate)
            # wave_ has to be inverted every second period to get a sinosoidal wave
            if i % 2 == 0:
                wave_ = self.amplitudes[1] * \
                    np.sin(np.pi * frequency * time_)
            else:
                wave_ = self.amplitudes[1] * \
                    np.sin(np.pi * frequency * time_) * -1
            self.modulation.extend(wave_)
        self.modulation = np.array(self.modulation)

    @property
    def duration(self):
        self._duration = self.wave.shape[0] / self.sample_rate
        return self._duration # alway include return for properties
     
    @property
    def wave_dot(self):
        self._wave_dot = np.gradient(self.wave, 1/self.sample_rate) # dx in seconds
        return self._wave_dot
    
    @property
    def peaks(self):
        self._peaks, _ = scipy.signal.find_peaks(self.wave, prominence=0.5)
        return self._peaks
    
    @property
    def troughs(self):
        self._troughs, _ = scipy.signal.find_peaks(-self.wave, prominence=0.5)
        return self._troughs
    
    def add_baseline_temp(self, baseline_temp):
        """
        Adds a baseline temperature to the wave. Should be around VAS = 35.

        Parameters
        ----------
        baseline_temp : float
            The baseline temperature to be added to the wave.
        """
        self.baseline_temp = baseline_temp
        self.wave = self.wave + self.baseline_temp

    def add_prolonged_peaks(self, time_to_be_added_per_peak, percetage_of_peaks):
        """
        Adds prolonged peaks to the wave.

        Parameters
        ----------
        time_to_be_added_per_peak : int
            The time to be added per peak.
        percetage_of_peaks : float
            The percentage of peaks to be prolonged.
        """
        peaks_chosen = self.rng_numpy.choice(self.peaks, int(
            len(self.peaks) * percetage_of_peaks), replace=False)
        wave_new = []
        for i in range(len(self.wave)):
            wave_new.append(self.wave[i])
            if i in peaks_chosen:
                wave_new.extend(
                    [self.wave[i]] * time_to_be_added_per_peak * self.sample_rate)
        self.wave = np.array(wave_new)

    def add_plateaus(self, plateau_duration, n_plateaus,
                     add_at_start="random", add_at_end=True):
        """
        Adds plateaus to the wave at random positions, but only when the temperature is rising 
        and the temperature is between the 25th and 75th percentile. The distance between the 
        plateaus is at least 1.5 times the plateau_duration.

        Parameters
        ----------
        plateau_duration : int
            The duration of the plateaus.
        n_plateaus : int
            The number of plateaus to be added.
        add_at_start : bool or str, optional
            If True, a plateau is added at the start of the wave. 
            If "random", it's randomly decided whether to add at the start (default is "random").
        add_at_end : bool or str, optional
            If True, a plateau is added at the end of the wave (default is True).

        Examples
        --------
        >>> stimuli.add_plateaus(plateau_duration=20, n_plateaus=4, add_at_start="random", add_at_end=True)
        """
        def _generate_plateau(start_value):
            """Generate a plateau with the given start value in °C."""
            return np.full(plateau_duration * self.sample_rate, start_value)
        
        def _to_bool(x):
            """If x is "random", it's converted to a random boolean."""
            return bool(self.rng.randint(0, 1)) if x == "random" else x

        for arg in [add_at_start, add_at_end]:
            if not isinstance(arg, (bool, str)) or (isinstance(arg, str) and arg != "random"):
                raise ValueError("add_at_start and add_at_end should be a boolean or 'random'.")
        add_at_start, add_at_end = _to_bool(add_at_start), _to_bool(add_at_end)

        # get indices where the temperature is rising and between the 25th and 75th percentile
        q25, q75 = np.percentile(self.wave, 25), np.percentile(self.wave, 75)
        idx_iqr_values = np.where((self.wave > q25) & (self.wave < q75) & (self.wave_dot > 0.07))[0] # [0] to get the indices

        # if add_at_start is False, remove indices that are within the first 10 seconds
        if not add_at_start:
            idx_iqr_values = idx_iqr_values[idx_iqr_values > 10 * self.sample_rate]
        # if add_at_end is False, remove indices that are within the last 10 seconds
        if not add_at_end:
            idx_iqr_values = idx_iqr_values[idx_iqr_values < len(self.wave) - 10 * self.sample_rate]

        # find indices for the random plateaus
        n_random_plateaus = n_plateaus - int(add_at_start) - int(add_at_end) 
        counter = 0
        while True: 
            counter += 1
            if counter > 1000:
                raise ValueError("""
                    Number and/or duration of plateaus is too high for the given wave (not enough
                    suitable index positions). It is recommended to always set add_at_end to True.
                    """)
            idx_plateaus = self.rng_numpy.choice(idx_iqr_values, n_random_plateaus, replace=False)
            if add_at_start:
                idx_plateaus = np.concatenate(([0], idx_plateaus))
            if add_at_end:
                idx_plateaus = np.concatenate((idx_plateaus, [len(self.wave)-1])) # -1 because of zero-indexing
            idx_plateaus = np.sort(idx_plateaus)
            # the distance between the plateaus should be at least 1.5 plateau_duration
            if np.all(np.diff(idx_plateaus) > 1.5 * plateau_duration * self.sample_rate):
                break # do-while loop in Python
                
        wave_new = []
        for i in range(len(self.wave)):
            wave_new.append(self.wave[i])
            if i in idx_plateaus:
                wave_new.extend(_generate_plateau(self.wave[i]))
        self.wave = np.array(wave_new)
    

def stimuli_extra(f, f_dot, sample_rate, s_RoC):
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
    s_RoC : float, optional
        The rate of change threshold (°C/s) for alternative labels.
        For more information about thresholds, also see: http://www.scholarpedia.org/article/Thermal_touch#Thermal_thresholds
    
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
    fig = go.Figure(
        layout=dict(
            xaxis=dict(title='Time (s)'),
            yaxis=dict(title='Temperature (°C) \ RoC (°C/s)')))

    fig.update_layout(
        autosize=False,
        height=300,
        width=900,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=10))
    
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
    print(f"Cooling segments [s] based on 'Label_alt' with a rate of change threshold of {s_RoC} (°C/s):\n")
    print((label_0_sizes/sample_rate).describe().apply('{:,.2f}'.format))

    return labels, labels_alt, fig


if __name__ == "__main__":
    pass