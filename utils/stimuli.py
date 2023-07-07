import math
import random
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt


class StimuliFunction():
    """
    The `StimuliFunction` class represent stimuli functions with sinusoidal waves and plateaus.


    Attributes
    ----------
    seed : int
        a random seed for generating random numbers.
    frequencies : numpy.array
        an array of frequencies for the sinusoidal waves.
    periods : numpy.array
        an array of periods of the sinusoidal waves, calculated as 1/frequency.
    amplitudes : numpy.array
        an array of amplitudes for the sinusoidal waves.
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
    random_phase : bool
        a flag to determine if the phase of the modulation is randomized.
    wave : numpy.array
        the stimuli function, calculated as the sum of the baseline and the modulation.
    wave_dot (@property): numpy.array
        the derivative of the stimuli function with respect to time (dx in seconds).
    peaks (@property): list
        a list of peak temperatures in the stimuli function (i.e. self.wave).

    Methods
    -------
    create_baseline():
        Creates the baseline sinusoidal wave. Class-internal method.
    create_modulation():
        Creates the modulation sinusoidal wave with varying frequency. Class-internal method.
    wave_dot():
        Calculates the derivative of the stimuli function with respect to time.
    add_baseline_temp(baseline_temp):
        Adds a baseline temperature to the stimuli function. 
        In a seperate function to be able to reuse seed for different baseline temperatures.
    add_prolonged_peaks(time_to_be_added_per_peak, percetage_of_peaks):
        Adds prolonged peaks to the stimuli function. Not used for now.
    add_plateaus(plateau_duration, n_plateaus, add_at_end=True):
        Adds plateaus to the stimuli function. 
        If add_at_end is True, a plateau is added at the end of the stimuli function.
    noise_that_sums_to_0(n, factor):
        Returns a noise vector that sums to 0 to be added to the period of the modulation if self.random_phase is True.

    Example
    -------
    ````python
    import numpy as np
    minimal_desired_duration = 200 # in seconds
    amplitudes = [1, 1.5] # the range will be 2 * sum(amplitudes)
    periods = [67, 10]  # 1 : 3 gives a good result
    frequencies = 1./np.array(periods)
    baseline_temp = 39.5 # with a calibrated pain threshold of 38 °C
    seed = 619

    stimuli = StimuliFunction(
        minimal_desired_duration, frequencies, amplitudes,
        random_phase=True, seed=seed)
    stimuli.add_baseline_temp(baseline_temp)
    stimuli.add_plateaus(plateau_duration=15, n_plateaus=4, add_at_end=True)
    ````

    Notes
    -----
    The usual range from pain threshold to pain tolerance is about 4 °C (without Capsaicin). \n
        
    From Andreas Strube's Neuron paper (2023) with stimuli of 8 s: \n
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

    def __init__(self, minimal_desired_duration, frequencies, amplitudes, random_phase=True, seed=None):
        """
        The constructor for StimuliFunction class.

        Parameters
        ----------
        minimal_desired_duration : float
            The minimal desired duration of the wave.
        frequencies : list
            The frequencies of the sinusoidal waves.
        amplitudes : list
            The amplitudes of the sinusoidal waves.
        random_phase : bool, optional
            If True, the phase of the modulation is randomized (default is True).
        seed : int, optional
            The seed for the random number generator (default is None, which generates a random seed).
        """
        # New instances of random number generators
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
        self.minimal_duration = math.ceil(
            self.minimal_desired_duration / self.periods[1]) * self.periods[1]
        # to get the real duration of the stimuli, always use self.wave.shape[0] / self.sample_rate
        self.sample_rate = 10

        # Additional variables
        self.modulation_n_periods = self.minimal_duration / self.periods[1]
        # if True, the phase of the modulation is randomized
        self.random_phase = random_phase

        # Summing self.baseline and self.modulation create the stimuli function self.wave
        self.create_baseline()
        self.create_modulation()
        self.wave = np.array(self.baseline) + np.array(self.modulation)
        self._duration = self.duration
        self._wave_dot = self.wave_dot
        self._peaks = self.peaks
        

    def create_baseline(self):
        time = np.arange(0, self.minimal_duration, 1/self.sample_rate)
        self.baseline = self.amplitudes[0] * np.sin(
            time * 2 * np.pi * self.frequencies[0])

    def create_modulation(self):
        # has to be created period-wise as the frequency varies with every period
        modulation_random_factor = self.noise_that_sums_to_0(
            n=int(self.modulation_n_periods),
            factor=0.6 if self.random_phase else 0)
        self.modulation = []
        for i in range(int(self.modulation_n_periods)):
            period = self.periods[1] + modulation_random_factor[i]
            frequency = 1/period
            time_ = np.arange(0, period, 1/self.sample_rate) # temporary time vector
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
        return self._duration # alway use return for properties
     
    @property
    def wave_dot(self):
        self._wave_dot = np.gradient(self.wave, 1/self.sample_rate) # dx in seconds
        return self._wave_dot
    
    @property
    def peaks(self):
        self._peaks, _ = scipy.signal.find_peaks(self.wave, prominence=0.5)
        return self._peaks
    
    def add_baseline_temp(self, baseline_temp):
        """
        Adds a baseline temperature to the wave.

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
                wave_new.extend([self.wave[i]] *
                                time_to_be_added_per_peak * self.sample_rate)
        self.wave = np.array(wave_new)

    def add_plateaus(self, plateau_duration, n_plateaus, add_at_end=True):
        """
        Adds plateaus to the wave. 
        Plateaus are added at random positions, but only when the temperature is rising 
        and the temperature is between the 25th and 75th percentile.

        Parameters
        ----------
        plateau_duration : int
            The duration of the plateaus.
        n_plateaus : int
            The number of plateaus to be added.
        add_at_end : bool, optional
            If True, a plateau is added at the end of the wave (default is True).
        """
        q25, q75 = np.percentile(self.wave, 25), np.percentile(self.wave, 75)
        # only for IQR temperature and only when temperature is rising
        idx_iqr_values = [
            i 
            for i in range(len(self.wave))
            if self.wave[i] > q25 and self.wave[i] < q75 and self.wave_dot[i] > 0.02
        ]
        # Subtract one from n_plateaus if add_at_end is True
        n_random_plateaus = n_plateaus - 1 if add_at_end else n_plateaus
        idx_plateaus = np.sort(self.rng_numpy.choice(
            idx_iqr_values, n_random_plateaus, replace=False))
        wave_new = []
        for i in range(len(self.wave)):
            wave_new.append(self.wave[i])
            if i in idx_plateaus:
                wave_new.extend([self.wave[i]] * plateau_duration * self.sample_rate)
        if add_at_end:
            plateau = np.full(plateau_duration * self.sample_rate, self.wave[-1])
            wave_new = np.concatenate((wave_new, plateau))
        self.wave = np.array(wave_new)

    def noise_that_sums_to_0(self, n, factor):
        """Returns noise vector to be added to the period of the modulation."""
        # create noise for n/2
        noise = self.rng_numpy.uniform(
            -factor * self.periods[1], 
            factor * self.periods[1],
            size = int(n/2))
        # double the noise with inverted values to sum to 0
        noise = np.concatenate((noise, -noise))
        # add 0 if length of n is odd
        if n % 2 == 1:
            noise = np.append(noise, 0)
        self.rng.shuffle(noise)
        return np.round(noise)
    

def stimuli_extra(f, f_dot, sample_rate, s_RoC=0.3):
    """
    For plotly graphing of f(x), f'(x), and labels.
    
    Parameters
    ----------
    f : array_like
        The function values at each time point.
    f_dot : array_like
        The derivative of the function at each time point.
    sample_rate : int
        The sample rate of the data.
    s_RoC : float, optional
        The rate of change threshold (°C/s) for alternative labels (default is 0.3).
    
    Returns
    -------
    labels : array_like
        A binary array where 0 indicates cooling and 1 indicates heating.
    labels_alt : array_like
        A ternary array where 0 indicates cooling, 1 indicates heating, and 2 indicates a rate of change less than s_RoC.
    fig : plotly.graph_objects.Figure
        The plotly figure.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    time = np.array(range(len(f))) / sample_rate
    
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
    
    # 0 for cooling, 1 for heating
    labels = (f_dot > 0).astype(int)
    # alternative: 0 for cooling, 1 for heating, 2 for RoC < s_RoC
    labels_alt = np.where(
        np.abs(f_dot) > s_RoC,
        labels, 2)
    
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

    return labels, labels_alt, fig


def cooling_segments(labels, sample_rate):
    '''Displays the number and length of cooling segments.'''
    """
    Displays the number and length of cooling segments.
    
    Parameters
    ----------
    labels : array_like
        A binary array where 0 indicates cooling and 1 indicates heating.
    sample_rate : int
        The sample rate of the stimuli function.
    
    Returns
    -------
    change : array_like
        The change of state in the labels, indicating when a transition between cooling and heating occurs.
    change_idx : array_like
        The indices where a change of state occurs.
    segments : dict
        A dictionary that is used to calculate the number and length of the cooling segments. 
        Usually not used, just printed.
    """
    from IPython.display import display

    change = np.concatenate([
        np.array([0]),  # as the sign cannot change with first value
        np.diff(labels > 0)*1], axis=0)

    # returns a list of indices where the conditions have been met
    change_idx = np.where(change == 1)[0]

    if labels[0] == 0:  # label we started with (cooling or heating)
        # in case 0 the first change_idx starts the first heating segment,
        # that is why we start with i=1::2 and not i=0::2
        # (we don't prepend / append any values from np.diff here)
        segments = {
            idx: np.diff(change_idx)[i::2] for idx, i in enumerate(list(range(2))[::-1])
        }
    elif labels[0] == 1:
        segments = {
            i: np.diff(change_idx)[i::2] for i in list(range(2))
        }

    # in seconds; only 1 column because jagged arrays can appear
    display(pd.DataFrame(
        {"Cooling segments [s]": segments[0]/sample_rate}
    ).describe().applymap('{:,.2f}'.format))

    return change, change_idx, segments


if __name__ == "__main__":
    pass