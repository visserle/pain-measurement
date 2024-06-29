import matplotlib.pyplot as plt
import numpy as np

from src.experiments.calibration.estimator import BayesianEstimatorVAS


def plot_trials(
    estimator: BayesianEstimatorVAS,
    interactive: bool = False,
) -> None:
    # Extract data from the estimator
    priors = estimator.priors
    likelihoods = estimator.likelihoods
    posteriors = estimator.posteriors
    range_temperature = estimator.range_temp
    min_temperature = estimator.minumum
    max_temperature = estimator.maximum
    trials = estimator.trials

    if interactive:
        # reload vs window if plot appears twice
        import ipywidgets as widgets
        from IPython.display import display

        def plot_trial(trial):
            plt.clf()
            plt.plot(range_temperature, priors[trial], label="Prior")
            plt.plot(range_temperature, likelihoods[trial], label="Likelihood")
            plt.plot(range_temperature, posteriors[trial], label="Posterior")
            plt.title(f"Trial {trial+1} of {trials}")
            plt.xlim([min_temperature, max_temperature])
            plt.ylim([0, 1])
            plt.xlabel("Temperature (°C)")
            plt.ylabel("Probability")
            plt.xticks(np.arange(min_temperature, max_temperature + 1, 1))
            plt.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=3)

        # Create a slider for the trial number
        trial_slider = widgets.IntSlider(min=0, max=trials - 1, step=1, value=0)
        # Create Next and Previous buttons,
        # define click events and link them to the buttons
        next_button = widgets.Button(description="Next")
        prev_button = widgets.Button(description="Previous")

        def next_button_clicked(b):  # b is the button instance
            if trial_slider.value < trial_slider.max:
                trial_slider.value += 1

        def prev_button_clicked(b):
            if trial_slider.value > trial_slider.min:
                trial_slider.value -= 1

        next_button.on_click(next_button_clicked)
        prev_button.on_click(prev_button_clicked)

        # Interact function to automatically update the plot when the slider is moved
        out = widgets.interactive_output(plot_trial, {"trial": trial_slider})
        # Display the slider and the buttons on top of the figure (box on box)
        display(
            widgets.VBox([widgets.HBox([trial_slider, prev_button, next_button]), out])
        )

    else:
        # One figure with all trials
        fig, ax = plt.subplots(trials, 1, figsize=(5, estimator.trials * 1.7))
        fig.suptitle(f"Calibration of VAS {estimator.vas_value}\n", fontsize=14)
        for trial in range(trials):
            ax[trial].plot(range_temperature, priors[trial], label="Prior")
            ax[trial].plot(range_temperature, likelihoods[trial], label="Likelihood")
            ax[trial].plot(range_temperature, posteriors[trial], label="Posterior")
            ax[trial].set_title(f"Trial {trial+1}")
            ax[trial].set_xlim([min_temperature, max_temperature])
            ax[trial].set_xticks(np.arange(min_temperature, max_temperature + 1, 1))
            ax[trial].set_ylim([0, 1])
            ax[trial].set_yticks([0, 1])

            # Add x-axis label to the last diagram only
            if trial == trials - 1:
                ax[trial].set_xlabel("Temperature (°C)")
        plt.legend(bbox_to_anchor=(0.5, -0.5), loc="upper center", ncol=3)
        plt.tight_layout()
        plt.show()
