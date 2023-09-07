# work in progress

# TODO
# - add logging
# - add doc


"""
Notes
-----
See calibration notebook for more details and visualizations.
"""

import math
import logging
import numpy as np
from scipy import stats

from .logger import setup_logger

logger = setup_logger(__name__.rsplit(".", maxsplit=1)[-1], level=logging.INFO)


class BayesianEstimatorVAS:
    """
    Recursive Bayesian estimation to:
    1. update belief about the temperature for the specific VAS value and
    2. decide on the temperature for the next trial.

    Example
    -------
    ```python
    from src.experiments.calibration import BayesianEstimatorVAS

    # Get estimate for VAS 0 (pain threshold)
    temp_start_vas0 = 38.
    trials = 7
    estimator_vas0 = BayesianEstimatorVAS(
        vas_value= 0, 
        temp_start=temp_start_vas0, 
        temp_std=3.5,
        trials=trials)

    for trial in range(estimator_vas0.trials):
        response = input(f'Is this stimulus painful? (y/n) ') 
        estimator_vas0.conduct_trial(response,trial=trial)
        
    # Get estimate for VAS 70
    temp_start_vas70 = estimator_vas0.get_estimate() + 3
    estimator_vas70 = BayesianEstimatorVAS(
        vas_value=70, 
        temp_start=temp_start_vas70,
        temp_std=1, # smaller std for higher temperatures
        trials=5)

    for trial in range(estimator_vas70.trials):
        response = input(f'Is this stimulus a 7 out of 10 (strong pain)? (y/n) ') 
        estimator_vas70.conduct_trial(response,trial=trial)
    
    ```

    Side notes
    ----------
    This code could be extended into a Kalman filter.
    For some intution on Kalman filters, see
    https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/ and
    https://praveshkoirala.com/2023/06/13/a-non-mathematical-introduction-to-kalman-filters-for-programmers/.
    
    _________________________________________________________________\n
    The usual range from pain threshold to pain tolerance is about 4 °C. \n
        
    From Andreas Strube's Neuron paper (2023) with stimuli of 8 s and Capsaicin: \n
    "During experiment 1, pain levels were calibrated to achieve \n
    - VAS10 (M = 38.1°C, SD = 3.5°C, Min = 31.8°C, Max = 44.8°C),
    - VAS30 (M = 39°C, SD = 3.5°C, Min = 32.2°C, Max = 45.3°C),
    - VAS50 (M = 39.9°C, SD = 3.6°C, Min = 32.5°C, Max = 46.2°C) and
    - VAS70 (M = 40.8°C, SD = 3.8°C, Min = 32.8°C, Max = 47.2°C) pain levels
    - (for highly effective conditioning, test trials, weakly effective conditioning and VAS70 pain stimulation, respectively).\n
    During experiment 2, pain levels were calibrated to achieve \n
    - VAS10 (M = 38.2°C, SD = 3.1°C, Min = 31.72°C, Max = 44.5°C)
    - VAS30 (M = 39°C, SD = 3.1°C, Min = 32.1°C, Max = 45.3°C),
    - VAS50 (M = 38.19°C, SD = 3.1°C, Min = 32.5°C, Max = 46.1°C) and
    - VAS70 (M = 40.5°C, SD = 3.2°C, Min = 32.8°C, Max = 46.9°C) pain levels
    - (for highly effective conditioning, test trials, weakly effective conditioning and pain stimulation, respectively)."
    """
    
    MAX_TEMP = 48.

    def __init__(self, vas_value, temp_start=38, temp_std=3.5, trials=7, likelihood_std=1, reduction_factor=0.95):
        """
        Initialize the VAS_Estimator object for recursive Bayesian estimation of temperature based on VAS values.

        Parameters
        ----------
        vas_value : int or float
            The Visual Analog Scale (VAS) value for which the temperature is to be estimated. 
            VAS is usually a scale from 0 to 100, where 0 indicates no pain and 100 indicates extreme pain.

        temp_start : float, optional
            The starting temperature in degrees Celsius for the estimation process. 
            Default is 38 degrees Celsius for VAS 0 (pain threshold) with Capsaicin.

        temp_std : float, optional
            The standard deviation of the initial Gaussian prior distribution for the temperature.
            Default is 3.5 degrees Celsius.
            
        trials : int, optional
            The number of trials to conduct for estimating the temperature.
            Default is 7.

        likelihood_std : float, optional
            The standard deviation of the likelihood function used in Bayesian updating.
            Default is 1.

        reduction_factor : float, optional
            The factor by which the standard deviation of the likelihood function is reduced after each trial.
            This allows the model to become more confident in its estimates as more data is collected.
            Default is 0.95.

        Attributes
        ----------
        range_temp : numpy.ndarray
            The range of temperatures considered for estimation, generated based on the temp_start and temp_std.

        prior : numpy.ndarray
            The initial prior probability distribution over the range of temperatures.

        current_temp : float
            The current best estimate of the temperature based on the maximum a posteriori (MAP) of the prior distribution.
            The current temperate cannot exceed the MAX_TEMP.

        temperatures : list
            List to store the temperatures considered in each trial.

        priors : list
            List to store the prior distributions used in each trial.

        likelihoods : list
            List to store the likelihood functions used in each trial.

        posteriors : list
            List to store the posterior distributions obtained in each trial.
        """
        self.vas_value = vas_value
        self.temp_start = temp_start
        self.temp_std = temp_std
        self.likelihood_std = likelihood_std
        self.reduction_factor = reduction_factor
        self.trials = trials

        # Define the range of temperatures to consider
        min_temp = self.temp_start - math.ceil(self.temp_std * 1.5)
        max_temp = self.temp_start + math.ceil(self.temp_std * 1.5)
        num = int((max_temp - min_temp) / 0.1) + 1
        self.range_temp = np.linspace(min_temp, max_temp, num)
        
        self.prior = stats.norm.pdf(self.range_temp, loc=self.temp_start, scale=self.temp_std)
        self.prior /= np.sum(self.prior)  # normalize

        self._current_temp = self.temp_start
        
        self.temps = [self.current_temp]
        self.priors = []
        self.likelihoods = []
        self.posteriors = []
    
    # Make sure the current temperature is never above MAX_TEMP
    @property
    def current_temp(self):
        return min(self._current_temp, self.MAX_TEMP)

    @current_temp.setter
    def current_temp(self, value):
        self._current_temp = value
        if self._current_temp > self.MAX_TEMP:
            logger.warning("Maximum temperature of %s °C reached.", self.MAX_TEMP)

    @property
    def steps(self):
        return np.diff(self.temps)

    def conduct_trial(self, response, trial=None):
        # Collect the subject's response and define a cdf likelihood function based on it
        if response == 'y':
            logger.info("Calibration trial (%s/%s): %s °C was over VAS %s.", trial+1, self.trials, self.current_temp, self.vas_value)
            likelihood = 1 - stats.norm.cdf(self.range_temp, loc=self.current_temp, scale=self.likelihood_std)
        else:
            logger.info("Calibration trial (%s/%s): %s °C was under VAS %s.", trial+1, self.trials, self.current_temp, self.vas_value)
            likelihood = stats.norm.cdf(self.range_temp, loc=self.current_temp, scale=self.likelihood_std)
        
        # Decrease the standard deviation of the likelihood function as we gain more information
        self.likelihood_std *= self.reduction_factor

        # Update the prior distribution with the likelihood function to get a posterior distribution
        posterior = likelihood * self.prior
        posterior /= np.sum(posterior)  # normalize

        # Choose the temperature for the next trial based on the posterior distribution
        self.current_temp = np.round(self.range_temp[np.argmax(posterior)], 1)

        # Store the distributions and temperature
        self.priors.append(self.prior)
        self.likelihoods.append(likelihood)
        self.posteriors.append(posterior)
        self.temps.append(self.current_temp)

        # Update the prior for the next iteration
        self.prior = np.copy(posterior)

        if trial == self.trials - 1: # last trial
            logger.info("Calibration estimate for VAS %s: %s °C.", self.vas_value, self.get_estimate())
            logger.info("Calibration steps were (°C): %s.", self.steps)
            # warning if all steps were in the same direction
            if np.all(np.diff(self.steps) >= 0) or np.all(np.diff(self.steps) < 0):
                logger.warning("Calibration steps were all in the same direction.")
   
    def get_estimate(self):
        return self.temps[-1]



# not used for now, as we use the BayesianEstimatorVAS instead

# class PainRegressor:
#     """
    
#     Notes
#     -----
#     Renamed from Psychometric-perceptual scaling to PainRegressor.
#     """
#     def __init__(self, pain_threshold):
#         self.pain_threshold = np.array(pain_threshold)
#         self.fixed_temperatures = np.array(pain_threshold + [0.5, 1, 2, 1])
#         self.vas_targtes = np.array([10, 30, 90])

#         self.vas_ratings = [0]
#         self.temps = [pain_threshold]

#         self.slope = None
#         self.intercept = None
            

#     def create_regression_(self, vas_rating, fixed_temperatures):
#         self.vas_ratings.append(vas_rating)
#         self.temps.append(fixed_temperatures)
#         logger.info("Calibration psychometric-perceptual scaling: %s °C was rated %s on the VAS scale.", fixed_temperatures, vas_rating)

#         if len(self.temps) == len(self.fixed_temperatures) + 1: # last trial for first regression
#             self.slope, self.intercept = np.polyfit(self.temps, self.vas_ratings, 1)
#             # y_pred = slope * x + intercept

#     def t():
#         pass
#         # x = (y_pred - intercept) / slope
#         # 10 30 90...

#     def improve_regression():
#         pass
