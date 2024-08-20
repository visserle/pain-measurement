from expyriment import stimuli

from src.experiments.measurement.rate_limiter import RateLimiter
from src.experiments.utils import scale_1d_value, scale_2d_tuple

CONSTANTS = {
    "screen_refresh_rate": 60,  # for 60 Hz monitors
    "sample_rate": 1000,
}

DEFAULTS = {
    "bar_length": 800,
    "bar_thickness": 30,
    "bar_position": (0, 0),
    "slider_width": 10,
    "slider_height": 90,
    "slider_color": (194, 24, 7),
    "slider_initial_position": (0, 0),
    "label_text_size": 40,
    "label_text_box_size": [250, 100],
}
DEFAULTS.update(CONSTANTS)


class VisualAnalogueScale:
    def __init__(
        self,
        experiment: object,
        config: dict | None = None,
    ):
        if config is None:
            config = {}
        self.config = {**DEFAULTS, **config}
        self.validate_config()

        self.experiment = experiment
        self.screen_size = experiment.screen.size
        self.rate_limiter = RateLimiter(self.config["sample_rate"], use_intervals=True)
        self.rate_limiter_screen = RateLimiter(self.config["screen_refresh_rate"])

        self.extract_and_scale_config()
        self._create_slider_elements()

        # Initialize the x position and the rating
        self.x_pos = None
        self.rating = None

    def validate_config(self):
        invalid_keys = [key for key in self.config if key not in DEFAULTS]
        if invalid_keys:
            raise ValueError(f"Invalid key(s) in config: {', '.join(invalid_keys)}")

    def extract_and_scale_config(self):
        for key, value in self.config.items():
            if isinstance(value, (tuple, list)) and "color" not in key:
                setattr(self, key, scale_2d_tuple(value, self.screen_size))
            elif isinstance(value, (int, float)):
                setattr(self, key, scale_1d_value(value, self.screen_size))
            else:
                setattr(self, key, value)  # directly assign if no scaling is needed

        # Additional calculations based on scaled values
        self.slider_min_x = -(self.bar_length / 2)
        self.slider_max_x = self.bar_length / 2
        self.label_right_position = (
            self.slider_max_x,
            self.bar_position[1] - scale_1d_value(110, self.screen_size),
        )
        self.label_left_position = (
            self.label_right_position[0] - self.bar_length,
            self.label_right_position[1],
        )

    def _create_slider_elements(self):
        # Create the bar, ends, slider and labels
        self.bar = stimuli.Rectangle(
            (self.bar_length, self.bar_thickness), position=self.bar_position
        )
        self.bar_end_left = stimuli.Rectangle(
            (5, self.bar_thickness * 3),
            position=(self.slider_min_x, self.bar_position[1]),
        )
        self.bar_end_right = stimuli.Rectangle(
            (5, self.bar_thickness * 3),
            position=(self.slider_max_x, self.bar_position[1]),
        )
        self.slider = stimuli.Rectangle(
            (self.slider_width, self.slider_height),
            position=self.slider_initial_position,
            colour=self.slider_color,
        )
        self.label_left = stimuli.TextBox(
            "Keine\nSchmerzen",
            size=self.label_text_box_size,
            position=self.label_left_position,
            text_size=self.label_text_size,
            text_font="timesnewroman",
        )
        self.label_right = stimuli.TextBox(
            "Sehr starke\nSchmerzen",
            size=self.label_text_box_size,
            position=self.label_right_position,
            text_size=self.label_text_size,
            text_font="timesnewroman",
        )

        # Preload stimuli for efficiency (OpenGL compression needs to be inhibited)
        for stimulus in [
            self.bar,
            self.bar_end_left,
            self.bar_end_right,
            self.slider,
            self.label_left,
            self.label_right,
        ]:
            stimulus.preload(inhibit_ogl_compress=True)

    def rate(
        self,
        instruction_textbox: stimuli.TextBox | None = None,
        timestamp: int | None = None,
    ) -> None:
        """
        Rate the stimulus by moving the slider and present the stimuli composition.

        Note that the optional instruction_textbox should be preloaded for performance
        reasons.
        """
        # Use the provided timestamp if given, otherwise, retrieve from experiment
        if timestamp is None:
            timestamp = self.experiment.clock.time

        # Rate the stimulus if allowed
        if self.rate_limiter.is_allowed(timestamp):
            # Update the slider position and rating
            self.x_pos = self.experiment.mouse.position[0]
            self.slider_x = max(min(self.x_pos, self.slider_max_x), self.slider_min_x)
            self.rating = round(
                (
                    (self.slider_x - self.slider_min_x)
                    / (self.slider_max_x - self.slider_min_x)
                    * 100
                ),
                3,  # round to 3 decimal places (= max precision for screen resolution)
            )

            # Uncomment to print the rating
            # print(f"Rating: {self.rating}")

        # Present the stimuli composition if allowed
        if self.rate_limiter_screen.is_allowed(timestamp):
            composition = stimuli.BlankScreen()
            self.slider.position = (self.slider_x, 0)
            stimuli_list = [
                self.bar,
                self.bar_end_left,
                self.bar_end_right,
                self.slider,
                self.label_left,
                self.label_right,
            ]
            # Add optional textbox if provided (OpenGL must be inhibited)
            if instruction_textbox:
                stimuli_list.append(instruction_textbox)
            # Plot all stimuli
            for stimulus in stimuli_list:
                stimulus.plot(composition)
            composition.present()


if __name__ == "__main__":
    from expyriment import control, design, stimuli

    # Initialize the experiment
    control.defaults.window_size = (800, 600)
    control.set_develop_mode(True)
    exp = design.Experiment()
    control.initialize(exp)

    # Default VAS configuration
    config = {}
    vas_slider = VisualAnalogueScale(exp, config)

    # Start the experiment
    control.start(skip_ready_screen=True)

    # Rating loop
    exp.clock.wait(3000, callback_function=lambda: vas_slider.rate())

    # End the experiment
    control.end()
