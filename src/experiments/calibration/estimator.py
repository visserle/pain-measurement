"""Baysian estimation of pain VAS value.

See calibration notebook for more details and visualizations."""

import logging

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


class BayesianEstimatorVAS:
    """
    Implements a Recursive Bayesian Estimator to:
    1. Continually update beliefs regarding the temperature corresponding to a specific
       VAS value.
    2. Decide the temperature for the succeeding trial based on the updated belief.

    Methods
    -------
    conduct_trial(response: str, trial: int) -> None:
        Conducts a single estimation trial and updates internal states based on the
        response.

    get_estimate() -> float:
        Retrieves the final estimated temperature after all trials are conducted.

    Example
    -------
    ```python
    import logging  # Set logging level to DEBUG for detailed output
    from src.expyriment.estimator import BayesianEstimatorVAS

    # Get estimate for VAS 50
    estimator_vas50 = BayesianEstimatorVAS(
        vas_value=50,
        trials=7,
        temp_start=40.0,
        temp_std=3.5,
        likelihood_std=1.0,
    )

    for trial in range(estimator_vas50.trials):
        response = input(f"Is this stimulus painful? (y/n) ")
        estimator_vas50.conduct_trial(response, trial=trial)
    ```
    """

    MAX_TEMP = 48.0

    def __init__(
        self,
        vas_value: int | float,
        trials: int,
        temp_start: float,
        temp_std: float,
        likelihood_std: float,
        reduction_factor: float = 0.85,
    ):
        """
        Initialize the VAS_Estimator object for recursive Bayesian estimation of
        temperature based on VAS values.

        Parameters
        ----------
        vas_value : int or float
            VAS value to be estimated.

        trials : int
            Number of trials for the estimation process.

        temp_start : float
            Initial temperature for estimation in degrees Celsius.
            Defaults to 38 degrees Celsius for VAS 0 (pain threshold) with Capsaicin.

        temp_std : float
            Standard deviation of the initial Gaussian prior distribution for
            temperature.

        likelihood_std : float
            Standard deviation of the likelihood function used in Bayesian updating.

        reduction_factor : float, optional, default=0.85
            Factor to reduce the standard deviation of the likelihood function after
            each trial. This allows the model to become more confident in its estimates
            as more data is collected.

        Attributes
        ----------
        prior : np.ndarray
            Initial prior probability distribution over the range of temperatures.

        current_temp : float
            Current best estimate of the temperature.
            If the current temperate exceeds the MAX_TEMP, a warning is logged.

        temps, priors, likelihoods, posteriors : list
            Lists to store temperature, prior distributions, likelihood functions, and
            posterior distributions for each trial, respectively.
        """
        self.vas_value = vas_value
        self.trials = trials
        self.temp_start = round(temp_start, 1)
        self.temp_std = temp_std
        self.likelihood_std = likelihood_std
        self.reduction_factor = reduction_factor

        # Define the range of temperatures to consider
        self.minumum = 38.0
        self.maximum = 49.9
        num = int((self.maximum - self.minumum) / 0.1) + 1
        self.range_temp = np.linspace(self.minumum, self.maximum, num)

        self.prior = stats.norm.pdf(
            self.range_temp,
            loc=self.temp_start,
            scale=self.temp_std,
        )
        self.prior /= np.sum(self.prior)  # normalize

        self._current_temp = self.temp_start

        self.temps = [self.current_temp]
        self.priors = []
        self.likelihoods = []
        self.posteriors = []

    @property
    def current_temp(self) -> float:
        return self._current_temp

    @current_temp.setter
    def current_temp(self, value):
        self._current_temp = value
        if self._current_temp >= self.MAX_TEMP:
            logger.warning(f"Estimated temperature exceeds {self.MAX_TEMP} °C.")

    @property
    def steps(self) -> np.ndarray:
        return np.diff(self.temps)

    def conduct_trial(
        self,
        response: str,
        trial: int,
    ) -> None:
        """
        Conducts a single estimation trial and updates internal states based on the
        response.

        Parameters
        ----------
        response : str
            Subject's response to the trial stimulus. Must be either "y" or "n" for
            painful or non-painful, respectively.

        trial : int
            Trial number (0-indexed).
        """
        # Collect the subject's response and define a cdf likelihood function
        trial_type = "over" if response == "y" else "under"
        logger.info(
            f"Calibration trial ({trial + 1}/{self.trials}): "
            f"{self.current_temp} °C was {trial_type} VAS {self.vas_value}."
        )

        if response == "y":
            likelihood = 1 - stats.norm.cdf(
                self.range_temp, loc=self.current_temp, scale=self.likelihood_std
            )
        else:
            likelihood = stats.norm.cdf(
                self.range_temp, loc=self.current_temp, scale=self.likelihood_std
            )

        # Decrease the standard deviation of the likelihood function as we gain
        # more information
        self.likelihood_std *= self.reduction_factor

        # Update the prior distribution with the likelihood function to get a
        # posterior distribution
        posterior = likelihood * self.prior
        posterior /= np.sum(posterior)  # normalize

        # Choose the temperature for the next trial based on the posterior distribution
        self.current_temp = np.round(self.range_temp[np.argmax(posterior)], 1).item()

        # Store the distributions and temperature
        self.priors.append(self.prior)
        self.likelihoods.append(likelihood)
        self.posteriors.append(posterior)
        self.temps.append(self.current_temp)

        # Update the prior for the next iteration
        self.prior = np.copy(posterior)

        if trial == self.trials - 1:  # last trial
            self._log_results()

    def _log_results(self) -> None:
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
            logger.warning(
                "Calibration steps for VAS %s were all in the same direction.",
                self.vas_value,
            )

    def validate_steps(self) -> bool:
        """
        Validates whether the temperature steps were all in the same direction,
        which is a sign of a bad estimate.

        True if the steps are not all in the same direction, False otherwise.
        """
        return ~(np.all(self.steps >= 0) or np.all(self.steps <= 0))

    def get_estimate(self) -> float:
        return self.temps[-1]
