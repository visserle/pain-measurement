"""Baysian estimation of pain VAS value. 

See calibration notebook for more details and visualizations."""

import logging
import math

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class BayesianEstimatorVAS:
    """
    Implements a Recursive Bayesian Estimator to:
    1. Continually update beliefs regarding the temperature corresponding to a specific VAS value.
    2. Decide the temperature for the succeeding trial based on the updated belief.

    This class has a class attribute MAX_TEMP, which is set to 47°C (maximum temperature considered by the estimator).

    Methods
    -------
    conduct_trial(response: str, trial: Optional[int]) -> None:
        Conducts a single estimation trial and updates internal states based on the response.

    get_estimate() -> float:
        Retrieves the final estimated temperature after all trials are conducted.

    Example
    -------
    ```python
    from src.expyriment.estimator import BayesianEstimatorVAS

    # Get estimate for VAS 50
    temp_start_vas50 = 40.0
    trials = 7
    estimator_vas50 = BayesianEstimatorVAS(
        vas_value=50, temp_start=temp_start_vas50, temp_std=3.5, trials=trials
    )

    for trial in range(estimator_vas50.trials):
        response = input(f"Is this stimulus painful? (y/n) ")
        estimator_vas50.conduct_trial(response, trial=trial)

    print(f"Estimated temperature for VAS 50: {estimator_vas50.get_estimate()} °C")
    ```

    Insights
    --------
    The usual temperature range between pain threshold and pain tolerance is around 3-4°C without Capsaicin.

    From Andreas Strube's Neuron paper (2023) with stimuli of 8 s and with Capsaicin: \n
    "During experiment 1, pain levels were calibrated to achieve \n
    - VAS10 (M = 38.1°C, SD = 3.5°C, Min = 31.8°C, Max = 44.8°C),
    - VAS30 (M = 39°C, SD = 3.5°C, Min = 32.2°C, Max = 45.3°C),
    - VAS50 (M = 39.9°C, SD = 3.6°C, Min = 32.5°C, Max = 46.2°C) and
    - VAS70 (M = 40.8°C, SD = 3.8°C, Min = 32.8°C, Max = 47.2°C) pain levels
    During experiment 2, pain levels were calibrated to achieve \n
    - VAS10 (M = 38.2°C, SD = 3.1°C, Min = 31.72°C, Max = 44.5°C)
    - VAS30 (M = 39°C, SD = 3.1°C, Min = 32.1°C, Max = 45.3°C),
    - VAS50 (M = 38.19°C, SD = 3.1°C, Min = 32.5°C, Max = 46.1°C) and
    - VAS70 (M = 40.5°C, SD = 3.2°C, Min = 32.8°C, Max = 46.9°C) pain levels
    """

    MAX_TEMP = 47.0

    def __init__(
        self,
        vas_value,
        temp_start,
        temp_std=3.5,
        trials=7,
        likelihood_std=1,
        reduction_factor=0.95,
    ):
        """
        Initialize the VAS_Estimator object for recursive Bayesian estimation of temperature based on VAS values.

        Parameters
        ----------
        vas_value : int or float
            VAS value to be estimated.

        temp_start : float, optional, default=38
            Initial temperature for estimation in degrees Celsius.
            Defaults to 38 degrees Celsius for VAS 0 (pain threshold) with Capsaicin.

        temp_std : float, optional, default=3.5
            Standard deviation of the initial Gaussian prior distribution for temperature.

        trials : int, optional, default=7
            Number of trials for the estimation process.

        likelihood_std : float, optional, default=1
            Standard deviation of the likelihood function used in Bayesian updating.

        reduction_factor : float, optional, default=0.95
            Factor to reduce the standard deviation of the likelihood function after each trial.
            This allows the model to become more confident in its estimates as more data is collected.

        Attributes
        ----------
        range_temp : np.ndarray
            The range of temperatures considered for estimation, generated based on the temp_start and temp_std.

        prior : np.ndarray
            Initial prior probability distribution over the range of temperatures.

        current_temp : float
            Current best estimate of the temperature.
            The current temperate cannot exceed the MAX_TEMP.

        temps, priors, likelihoods, posteriors : list
            Lists to store temperature, prior distributions, likelihood functions, and posterior distributions for each trial, respectively.
        """
        self.vas_value = vas_value
        self.temp_start = temp_start
        self.temp_std = temp_std
        self.likelihood_std = likelihood_std
        self.reduction_factor = reduction_factor
        self.trials = trials

        # Define the range of temperatures to consider
        self.min_temp = self.temp_start - math.ceil(self.temp_std * 1.5)
        self.max_temp = self.temp_start + math.ceil(self.temp_std * 1.5)
        num = int((self.max_temp - self.min_temp) / 0.1) + 1
        self.range_temp = np.linspace(self.min_temp, self.max_temp, num)

        self.prior = stats.norm.pdf(
            self.range_temp, loc=self.temp_start, scale=self.temp_std
        )
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
        if self._current_temp >= self.MAX_TEMP:
            logger.warning("Maximum temperature of %s °C reached.", self.MAX_TEMP)

    @property
    def steps(self):
        return np.diff(self.temps)

    def conduct_trial(self, response: str, trial: int) -> None:
        """
        Conducts a single estimation trial and updates internal states based on the response.

        Parameters
        ----------
        response : str
            Subject's response to the trial stimulus. Must be either "y" or "n".

        trial : int
            Trial number (0-indexed).
        """
        # Collect the subject's response and define a cdf likelihood function based on it
        if response == "y":
            logger.info(
                "Calibration trial (%s/%s): %s °C was over VAS %s.",
                trial + 1,
                self.trials,
                self.current_temp,
                self.vas_value,
            )
            likelihood = 1 - stats.norm.cdf(
                self.range_temp, loc=self.current_temp, scale=self.likelihood_std
            )
        else:
            logger.info(
                "Calibration trial (%s/%s): %s °C was under VAS %s.",
                trial + 1,
                self.trials,
                self.current_temp,
                self.vas_value,
            )
            likelihood = stats.norm.cdf(
                self.range_temp, loc=self.current_temp, scale=self.likelihood_std
            )

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

        if trial == self.trials - 1:  # last trial
            logger.info(
                "Calibration estimate for VAS %s: %s °C.",
                self.vas_value,
                self.get_estimate(),
            )
            logger.debug(
                "Calibration steps for VAS %s were (°C): %s.",
                self.vas_value,
                self.steps,
            )
            if not self.validate_steps():
                logger.error(
                    "Calibration steps for VAS %s were all in the same direction.",
                    self.vas_value,
                )

    def validate_steps(self) -> bool:
        """
        Validates whether the temperature steps were all in the same direction, which is a sign of a bad estimate.

        True if the steps are not all in the same direction, False otherwise.
        """
        return ~(np.all(self.steps >= 0) or np.all(self.steps <= 0))

    def get_estimate(self) -> float:
        return self.temps[-1]
