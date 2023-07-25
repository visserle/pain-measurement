# work in progress

# TODO
# - add logging

import numpy as np
from scipy import stats

class PainThresholdEstimator:
    """
    
    Example
    -------
    ```python
    pain_estimator = PainThresholdEstimator(mean_temperature=39)

    for trial in range(pain_estimator.trials):
        response = input(f'Is {pain_estimator.current_temperature} °C painful? (y/n) ')
        pain_estimator.conduct_trial(response)

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
        self._steps = self.steps
        self.priors = []
        self.likelihoods = []
        self.posteriors = []

    def conduct_trial(self, response):
        if response == 'y':
            likelihood = 1 - stats.norm.cdf(self.range_temperature, loc=self.current_temperature, scale=self.likelihood_std)
        else:
            likelihood = stats.norm.cdf(self.range_temperature, loc=self.current_temperature, scale=self.likelihood_std)

        self.likelihood_std *= self.reduction_factor

        posterior = likelihood * self.prior
        posterior /= np.sum(posterior)  # normalize

        self.current_temperature = np.round(self.range_temperature[np.argmax(posterior)], 1)

        self.priors.append(self.prior)
        self.likelihoods.append(likelihood)
        self.posteriors.append(posterior)
        self.temperatures.append(self.current_temperature)

        self.prior = np.copy(posterior)

    def get_estimate(self):
        return self.temperatures[-1]

    @property
    def steps(self):
        return np.diff(self.temperatures)
