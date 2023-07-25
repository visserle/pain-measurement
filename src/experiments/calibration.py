# work in progress

# TODO
# - add logging
# - add doc


"""
Notes
-----
See calibration notebook for more details and visualizations.
"""

import numpy as np
from scipy import stats
import logging

class PainThresholdEstimator:
    """
    

    Example
    -------
    ```python
    pain_estimator = PainThresholdEstimator(mean_temperature=39)

    for trial in range(pain_estimator.trials):
        response = input(f'Is {pain_estimator.current_temperature} °C painful? (y/n) ')
        pain_estimator.conduct_trial(response,trial=trial)

    print(f"Calibration pain threshold estimate: {pain_estimator.get_estimate()} °C")
    ```
    """
    def __init__(self, mean_temperature, std_temperature=4, likelihood_std=1, reduction_factor=0.9, trials=7):
        self.mean_temperature = mean_temperature
        self.std_temperature = std_temperature
        self.likelihood_std = likelihood_std
        self.reduction_factor = reduction_factor
        self.trials = trials

        min_temperature = self.mean_temperature - 6
        max_temperature = self.mean_temperature + 6
        num_steps = int((max_temperature - min_temperature) / 0.1) + 1
        self.range_temperature = np.linspace(min_temperature, max_temperature, num_steps)
        
        self.prior = stats.norm.pdf(self.range_temperature, loc=self.mean_temperature, scale=self.std_temperature)
        self.prior /= np.sum(self.prior)  # normalize

        self.current_temperature = np.round(self.range_temperature[np.argmax(self.prior)], 1)
        self.temperatures = [self.current_temperature]
        self.priors = []
        self.likelihoods = []
        self.posteriors = []

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
            self.logger.addHandler(handler)

    @property
    def steps(self):
        return np.diff(self.temperatures)

    def conduct_trial(self, response, trial=None):
        # Collect the subject's response and define a cdf likelihood function based on it
        if response == 'y':
            self.logger.info(f"Calibration pain threshold trial {trial+1}: {self.current_temperature} °C was painful.")
            likelihood = 1 - stats.norm.cdf(self.range_temperature, loc=self.current_temperature, scale=self.likelihood_std)
        else:
            self.logger.info(f"Calibration pain threshold trial {trial+1}: {self.current_temperature} °C was not painful.")
            likelihood = stats.norm.cdf(self.range_temperature, loc=self.current_temperature, scale=self.likelihood_std)
        
        # Decrease the standard deviation of the likelihood function as we gain more information
        self.likelihood_std *= self.reduction_factor

        # Update the prior distribution with the likelihood function to get a posterior distribution
        posterior = likelihood * self.prior
        posterior /= np.sum(posterior)  # normalize

        # Choose the temperature for the next trial based on the posterior distribution
        self.current_temperature = np.round(self.range_temperature[np.argmax(posterior)], 1)

        # Store the distributions and temperature
        self.priors.append(self.prior)
        self.likelihoods.append(likelihood)
        self.posteriors.append(posterior)
        self.temperatures.append(self.current_temperature)

        # Update the prior for the next iteration
        self.prior = np.copy(posterior)

        if trial == self.trials - 1: # last trial
            self.logger.info(f"Calibration pain threshold steps (°C) were {self.steps}.\n")
            self.logger.info(f"Calibration pain threshold estimate: {self.get_estimate()} °C")

    def get_estimate(self):
        return self.temperatures[-1]

