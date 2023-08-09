# work in progress

# TODO
# - add logging
# - add doc


"""
Notes
-----
See calibration notebook for more details and visualizations.
"""

import logging
import numpy as np
from scipy import stats

from .logger import setup_logger

logger = setup_logger(__name__.rsplit(".", maxsplit=1)[-1], level=logging.INFO)


class PainThresholdEstimator:
    """
    Recursive Bayesian estimation to
        1. update belief about the heat pain threshold and
        2. decide on the temperature for the next trial.

    Example
    -------
    ```python
    pain_estimator = PainThresholdEstimator(mean_temperature=39)

    for trial in range(pain_estimator.trials):
        response = input(f'Is {pain_estimator.current_temperature} °C painful? (y/n) ')
        pain_estimator.conduct_trial(response,trial=trial)

    print(f"Calibration pain threshold estimate: {pain_estimator.get_estimate()} °C")
    ```

    Side notes
    ----------
    This code could be extended into a Kalman filter.
    For some intution on Kalman filters, see
    https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/ and
    https://praveshkoirala.com/2023/06/13/a-non-mathematical-introduction-to-kalman-filters-for-programmers/.
    """
    def __init__(self, mean_temperature, std_temperature=4, likelihood_std=1, reduction_factor=0.95, trials=7):   
        self.mean_temperature = mean_temperature
        self.std_temperature = std_temperature
        self.likelihood_std = likelihood_std
        self.reduction_factor = reduction_factor
        self.trials = trials

        # Define the range of temperatures to consider
        min_temperature = self.mean_temperature - 6
        max_temperature = self.mean_temperature + 6
        num = int((max_temperature - min_temperature) / 0.1) + 1
        self.range_temperature = np.linspace(min_temperature, max_temperature, num)
        
        self.prior = stats.norm.pdf(self.range_temperature, loc=self.mean_temperature, scale=self.std_temperature)
        self.prior /= np.sum(self.prior)  # normalize

        self.current_temperature = np.round(self.range_temperature[np.argmax(self.prior)], 1) # = mean_temperature
        self.temperatures = [self.current_temperature]
        self.priors = []
        self.likelihoods = []
        self.posteriors = []
    

    @property
    def steps(self):
        return np.diff(self.temperatures)

    def conduct_trial(self, response, trial=None):
        # Collect the subject's response and define a cdf likelihood function based on it
        if response == 'y':
            logger.info("Calibration pain threshold trial %s: %s °C was painful.", trial+1, self.current_temperature)
            likelihood = 1 - stats.norm.cdf(self.range_temperature, loc=self.current_temperature, scale=self.likelihood_std)
        else:
            logger.info("Calibration pain threshold trial %s: %s °C was not painful.", trial+1, self.current_temperature)
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
            logger.info("Calibration pain threshold steps (°C) were: %s.\n", self.steps)
            logger.info("Calibration pain threshold estimate: %s °C.", self.get_estimate())

    def get_estimate(self):
        return self.temperatures[-1]


class PainRegressor:
    """
    
    Notes
    -----
    Renamed from Psychometric-perceptual scaling to PainRegressor.
    """
    def __init__(self, pain_threshold):
        self.pain_threshold = np.array(pain_threshold)
        self.fixed_temperatures = np.array(pain_threshold + [0.5, 1, 2, 1])
        self.vas_targtes = np.array([10, 30, 90])

        self.vas_ratings = [0]
        self.temperatures = [pain_threshold]

        self.slope = None
        self.intercept = None
            

    def create_regression_(self, vas_rating, fixed_temperatures):
        self.vas_ratings.append(vas_rating)
        self.temperatures.append(fixed_temperatures)
        logger.info("Calibration psychometric-perceptual scaling: %s °C was rated %s on the VAS scale.", fixed_temperatures, vas_rating)

        if len(self.temperatures) == len(self.fixed_temperatures) + 1: # last trial for first regression
            self.slope, self.intercept = np.polyfit(self.temperatures, self.vas_ratings, 1)
            # y_pred = slope * x + intercept

    def t():
        pass
        # x = (y_pred - intercept) / slope
        # 10 30 90...

    def improve_regression():
        pass


if __name__ == "__main__":
    pass
