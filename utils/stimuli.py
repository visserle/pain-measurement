import math
import random
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display


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
    thetas : numpy.array
        an array of initial angles for the sinusoidal waves, default are zeros.
    baseline_temp : float
        the baseline temperature.
    desired_duration : float
        the desired duration for the stimuli function. 
    duration : float
        the duration for the vanilla stimuli function (without add_ methods), calculated as a multiple of the period of the modulation.
        always use self.wave.shape[0] / self.sample_rate to get the actual duration.
    sample_rate : int
        the sample rate for the stimuli function.
    modulation_n_periods : float
        the number of periods for the modulation.
    random_phase : bool
        a flag to determine if the phase of the modulation is randomized.
    wave : numpy.array
        the stimuli function, calculated as the sum of the baseline and the modulation.
    _wave_dot : numpy.array
        the derivative of the stimuli function with respect to time (dx in seconds).
    peaks : list
        a list of peak temperatures in the stimuli function (i.e. self.wave).

    Methods
    -------
    create_baseline():
        Creates the baseline sinusoidal wave.
    create_modulation():
        Creates the modulation sinusoidal wave with varying frequency.
    wave_dot():
        Calculates the derivative of the stimuli function with respect to time.
    add_prolonged_peaks(time_to_be_added_per_peak, percetage_of_peaks):
        Adds prolonged peaks to the stimuli function. Not used for now.
    add_plateaus(n_plateaus, duration_per_plateau):
        Adds plateaus to the stimuli function.
    noise_that_sums_to_0(n, factor):
        Returns a noise vector that sums to 0 to be added to the period of the modulation if self.random_phase is True.
    plot(wave, baseline_temp):
        Plots the stimuli function in plotly.
        
    Example
    -------
    ````python
    duration = 200 # in seconds
    amplitudes = [1, 1.5] # the range will be 2 * sum(amplitudes)
    periods = [67, 10]  # 1 : 3 gives a good result
    frequencies = 1./np.array(periods)
    baseline_temp = 39.5 # with a calibrated pain threshold of 38 °C

    stimuli = StimuliFunction(
        duration, frequencies, amplitudes, baseline_temp, 
        random_phase=True, seed=764)
    stimuli.add_plateaus(n_plateaus=4, duration_per_plateau=15)
    ````

    Notes
    -----
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
    def __init__(self, desired_duration, frequencies, amplitudes, baseline_temp=38, random_phase=True, seed=None):
        if seed is None:
            self.seed = np.random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed), np.random.seed(self.seed)

        # Sinusoidal waves where [0] is the baseline and [1] the modulation which gets added on top
        self.frequencies = np.array(frequencies)
        self.periods = 1/self.frequencies
        self.amplitudes = np.array(amplitudes)
        self.thetas = np.zeros(len(frequencies))  # not used for now
        self.baseline_temp = baseline_temp

        # Duration and sampling (without add_ methods)
        self.desired_duration = desired_duration
        # the "true" duration is a multiple of the period of the modulation
        self.duration = math.ceil(
            self.desired_duration / self.periods[1]) * self.periods[1]
        self.sample_rate = 10

        # Additional variables
        self.modulation_n_periods = self.duration / self.periods[1]
        # if True, the phase of the modulation is randomized
        self.random_phase = random_phase

        # Summing self.baseline and self.modulation create the stimuli function self.wave
        self.create_baseline()
        self.create_modulation()
        self.wave = np.array(self.baseline) + np.array(self.modulation)
        self._wave_dot = self.wave_dot
        self.peaks, _ = scipy.signal.find_peaks(
            self.wave, height=self.baseline_temp)

    def create_baseline(self):
        time = np.arange(0, self.duration, 1/self.sample_rate)
        self.baseline = self.amplitudes[0] * np.sin(
            time * 2 * np.pi * self.frequencies[0] + self.thetas[0]) + self.baseline_temp
        return self.baseline

    def create_modulation(self):
        # has to be created period-wise as the frequency varies with every period
        modulation_random_factor = self.noise_that_sums_to_0(
            n=int(self.modulation_n_periods),
            factor=0.6 if self.random_phase else 0)
        self.modulation = []
        for i in range(int(self.modulation_n_periods)):
            period = self.periods[1] + modulation_random_factor[i]
            frequency = 1/period
            time_temp = np.arange(0, 1/frequency, 1/self.sample_rate)
            # wave_temp has to be inverted every second period to get a sinosoidal wave
            if i % 2 == 0:
                wave_temp = self.amplitudes[1] * \
                    np.sin(np.pi * frequency * time_temp)
            else:
                wave_temp = self.amplitudes[1] * \
                    np.sin(np.pi * frequency * time_temp) * -1
            self.modulation.extend(wave_temp)
        self.modulation = np.array(self.modulation)
        return self.modulation
      
    @property
    def wave_dot(self):
        self._wave_dot = np.gradient(self.wave, 1/self.sample_rate) # dx in seconds
        return self._wave_dot

    def add_prolonged_peaks(self, time_to_be_added_per_peak, percetage_of_peaks):
        peaks_chosen = np.random.choice(self.peaks, int(
            len(self.peaks) * percetage_of_peaks), replace=False)
        wave_new = []
        for i in range(len(self.wave)):
            wave_new.append(self.wave[i])
            if i in peaks_chosen:
                wave_new.extend([self.wave[i]] *
                                time_to_be_added_per_peak * self.sample_rate)
        self.wave = np.array(wave_new)
        return self.wave

    def add_plateaus(self, n_plateaus, duration_per_plateau):
        q25, q75 = np.percentile(self.wave, 25), np.percentile(self.wave, 75)
        # only for IQR temp and only when temp is rising
        idx_iqr_values = [
            i 
            for i in range(len(self.wave))
            if self.wave[i] > q25 and self.wave[i] < q75 and self.wave_dot[i] > 0.02
        ]
        idx_plateaus = np.sort(np.random.choice(
            idx_iqr_values, n_plateaus, replace=False))
        wave_new = []
        for i in range(len(self.wave)):
            wave_new.append(self.wave[i])
            if i in idx_plateaus:
                wave_new.extend([self.wave[i]] * duration_per_plateau * self.sample_rate)
        self.wave = np.array(wave_new)
        return self.wave

    def noise_that_sums_to_0(self, n, factor):
        """Returns noise vector to be added to the period of the modulation."""
        # create noise for n/2
        noise = np.random.uniform(
            -factor * self.periods[1], 
            factor * self.periods[1],
            size = int(n/2))
        # double the noise with inverted values to sum to 0
        noise = np.concatenate((noise, -noise))
        # add 0 if length of n is odd
        if n % 2 == 1:
            noise = np.append(noise, 0)
        random.shuffle(noise)
        return np.round(noise)

    def plot(self, wave, baseline_temp=False):
        time = np.array(range(len(wave))) / self.sample_rate
        fig = go.Figure(
            go.Scatter(x=time, y=wave, line=dict(color='royalblue')),
            layout=dict(
                xaxis=dict(
                    title='Time (s)', tickmode='linear',
                    tick0=0, dtick=10),
                yaxis=dict(
                    title='Temperature (°C)'),
                    autosize=False,
                    height=300,
                    width=900,
                    margin=dict(l=20, r=20, t=20, b=20)))
        if baseline_temp:
            fig.add_hline(y=int(self.baseline_temp))
        return fig



def stimuli_extra(f, f_dot, time, s_RoC=0.5):
    """
    For plotly graphing of f(x), f'(x), and labels.
    
    Parameters
    ----------
    f : array_like
        The function values at each time point.
    f_dot : array_like
        The derivative of the function at each time point.
    time : array_like
        The time points.
    s_RoC : float, optional
        The rate of change threshold for alternative labels (default is 0.5).
    
    Returns
    -------
    labels : array_like
        A binary array where 0 indicates cooling and 1 indicates heating.
    labels_alt : array_like
        A ternary array where 0 indicates cooling, 1 indicates heating, and 2 indicates a rate of change less than s_RoC.
    fig : plotly.graph_objects.Figure
        The plotly figure.
    """
    fig = go.Figure(
        layout=dict(
            xaxis=dict(title='Time (s)'),
            yaxis=dict(title='Temperature (°C) \ RoC (°C/s)')))

    fig.update_layout(
        autosize=False,
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

    func = [f, f_dot, labels]
    func_names = "f(x)", "f'(x)", "Label"
    colors = 'royalblue', 'skyblue', 'springgreen'
    for idx, i in enumerate(func):
        fig.add_scatter(x=time, y=i, name=func_names[idx])
        fig.data[idx].line.color = colors[idx]

    fig.add_scatter(x=time, y=labels_alt,
                    name="Label (alt)", visible="legendonly")

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