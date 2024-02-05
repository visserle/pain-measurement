# TODO
# - find out if we need exp.mouse.track_motion_events = True or not -> probably not

from expyriment import stimuli

from src.expyriment.utils import scale_1d_value, scale_2d_tuple


class VisualAnalogueScale:
    def __init__(self, experiment, vas_config: dict):
        self.experiment = experiment
        self.screen_size = self.experiment.screen.size
        self.bar_length = scale_1d_value(vas_config.get("bar_length", 600), self.screen_size)
        self.bar_thickness = scale_1d_value(vas_config.get("bar_thickness", 30), self.screen_size)
        self.bar_position = scale_2d_tuple(vas_config.get("bar_position", (0, 0)), self.screen_size)
        self.slider_width = scale_1d_value(vas_config.get("slider_width", 10), self.screen_size)
        self.slider_height = scale_1d_value(vas_config.get("slider_height", 90), self.screen_size)
        self.slider_color = vas_config.get("slider_color", (194, 24, 7))
        self.slider_initial_position = scale_2d_tuple(
            vas_config.get("slider_initial_position", (0, self.bar_position[1])), self.screen_size
        )
        self.slider_min_x = -(self.bar_length / 2)
        self.slider_max_x = self.bar_length / 2
        self.mouse_sampling_rate = vas_config.get("mouse_sampling_rate", 60)
        self.mouse_refresh = max(1000 // self.mouse_sampling_rate, 1)
        self.label_text_size = scale_1d_value(
            vas_config.get("label_text_size", 40), self.screen_size
        )
        self.label_text_box_size = scale_2d_tuple(
            vas_config.get("label_text_box_size", (250, 100)), self.screen_size
        )
        self.label_right_position = vas_config.get(
            "label_position",
            (self.slider_max_x, self.bar_position[1] - scale_1d_value(100, self.screen_size)),
        )
        self.label_left_position = (
            self.label_right_position[0] - self.bar_length,
            self.label_right_position[1],
        )

        self.create_slider_elements()

        self.last_time = -1
        self.last_x_pos = -1

        self.rating = 50

    def create_slider_elements(self):
        # Create the bar, ends, and slider
        self.bar = stimuli.Rectangle(
            (self.bar_length, self.bar_thickness), position=self.bar_position
        )
        self.bar_end_left = stimuli.Rectangle(
            (5, self.bar_thickness * 3), position=(self.slider_min_x, self.bar_position[1])
        )
        self.bar_end_right = stimuli.Rectangle(
            (5, self.bar_thickness * 3), position=(self.slider_max_x, self.bar_position[1])
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
        )
        self.label_right = stimuli.TextBox(
            "Sehr starke\nSchmerzen",
            size=self.label_text_box_size,
            position=self.label_right_position,
            text_size=self.label_text_size,
        )

        # Preload stimuli for efficiency
        for stimulus in [
            self.bar,
            self.bar_end_left,
            self.bar_end_right,
            self.slider,
            self.label_left,
            self.label_right,
        ]:
            stimulus.preload(inhibit_ogl_compress=True)  # otherwise the text will be blurry

    def rate(self, instruction_textbox=None):
        # NOTE we can easily move the if statement outside of the method if needed in the future
        if (
            ((current_time := self.experiment.clock.time) % self.mouse_refresh == 0)
            and (current_time != self.last_time)  # TODO not needed with low refresh rate
            # and ((current_x_pos := self.experiment.mouse.position[0]) != last_x_pos) NOTE
        ):
            # Adjust slider position based on mouse X-coordinate within boundaries
            current_x_pos = self.experiment.mouse.position[0]
            slider_x = max(min(current_x_pos, self.slider_max_x), self.slider_min_x)
            rating = int(
                (slider_x - self.slider_min_x) / (self.slider_max_x - self.slider_min_x) * 100
            )

            # Create a composition to show multiple elements simultaneously
            composition = stimuli.BlankScreen()
            self.slider.position = (slider_x, 0)
            stimuli_list = [
                self.bar,
                self.bar_end_left,
                self.bar_end_right,
                self.slider,
                self.label_left,
                self.label_right,
            ]

            # Conditionally add the optional textbox
            if instruction_textbox:
                stimuli_list.append(instruction_textbox)

            # Plot all stimuli
            for stimulus in stimuli_list:
                stimulus.plot(composition)
            composition.present()

            # Update time and position
            self.last_time = current_time
            self.last_x_pos = current_x_pos

            self.rating = rating
