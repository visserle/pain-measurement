import re
import tkinter as tk

import toml
import yaml
from expyriment.stimuli import TextBox
from screeninfo import get_monitors

BASE_SCREEN_SIZE = (1920, 1200)


def load_configuration(file_path: str) -> dict:
    """Load configuration from a TOML file."""
    with open(file_path, "r", encoding="utf8") as file:
        return toml.load(file)


def load_script(file_path: str) -> dict:
    """Load script from a YAML file."""
    with open(file_path, "r", encoding="utf8") as file:
        return yaml.safe_load(file)


def prepare_script(
    script: dict,
    text_size: int,
    text_box_size: tuple[int, int],
    parent_key: str = None,
) -> None:
    """
    Recursively convert existings script strings to CustomTextBox stimuli and preload them.

    (With special preloading for 'instruction' as they are shown with the visual analogue scale composition)
    """
    for key, value in script.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries, passing down the current key as the parent_key
            prepare_script(value, text_size, text_box_size, key)
        else:
            # Convert strings to CustomTextBox stimuli
            script[key] = CustomTextBox(
                text=value,
                size=text_box_size,
                position=[0, 0],
                text_font="timesnewroman",
                text_size=text_size,
            )
            # Special preloading for 'instruction' which contains the visual analogue scale composition
            if parent_key == "instruction":
                script[key].preload(inhibit_ogl_compress=True)
            else:
                script[key].preload()


def _scale_ratio(
    screen_size: tuple[int, int],
    base_screen_size: tuple[int, int],
) -> float:
    """
    Calculate the scale ratio based on the screen size.
    """
    scale_ratio_width = screen_size[0] / base_screen_size[0]
    scale_ratio_height = screen_size[1] / base_screen_size[1]
    # Use the smaller ratio to ensure fit
    scale_ratio = min(scale_ratio_width, scale_ratio_height)
    return scale_ratio


def scale_1d_value(
    base_value: int | float,
    screen_size: tuple[int, int],
    base_screen_size: tuple[int, int] = BASE_SCREEN_SIZE,
) -> int | float:
    """
    Calculate the adjusted value based on the screen size for 1D values like length, width or text size.

    Parameters:
    - base_value: int or float, base value to scale from
    - screen_size: tuple, current screen size (width, height)
    - base_screen_size: tuple, base screen size (width, height) for scaling reference, default=(1920, 1200)

    Returns:
    - scaled_value: int or float, scaled value based on the current screen size
    """
    scale_factor = _scale_ratio(screen_size, base_screen_size)
    if isinstance(base_value, int):
        scaled_value = int(base_value * scale_factor)
    else:
        scaled_value = float(base_value * scale_factor)
    return scaled_value


def scale_2d_tuple(
    base_value: tuple[int, int],
    screen_size: tuple[int, int],
    base_screen_size: tuple[int, int] = BASE_SCREEN_SIZE,
) -> tuple[int, int]:
    """
    Calculate the adjusted value based on the screen size for 2D values like position or size.

    Parameters:
    - base_value: tuple, base value to scale from (width, height)
    - screen_size: tuple, current screen size (width, height)
    - base_screen_size: tuple, base screen size (width, height) for scaling reference, default=(1920, 1200)

    Returns:
    - scaled_value: tuple, scaled value based on the current screen size
    """
    scale_factor = _scale_ratio(screen_size, base_screen_size)
    scaled_value = (
        int(base_value[0] * scale_factor),
        int(base_value[1] * scale_factor),
    )
    return scaled_value


def center_tk_window(window: tk.Tk, primary_screen: bool = False) -> tuple[int, int]:
    """Center a window, by default on the first available non-primary screen if available, otherwise on the primary screen."""
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()

    # Get list of monitors and sort them
    monitors = sorted(get_monitors(), key=lambda m: m.is_primary)
    # Non-primary monitor comes first (False < True)
    if primary_screen:
        target_monitor = monitors[-1]
    else:
        target_monitor = monitors[0]

    screen_width = target_monitor.width
    screen_height = target_monitor.height
    center_x = int(target_monitor.x + (screen_width / 2 - width / 2))
    center_y = int(target_monitor.y + (screen_height / 2 - height / 2))

    # Move window to the center of the chosen monitor
    window.geometry(f"+{center_x}+{center_y}")

    return (center_x, center_y)


class CustomTextBox(TextBox):
    """
    Expyriment text box without leading whitespace stripping.

    This class is a copy of the TextBox class from expyriment.stimuli.textbox
    with the only difference that it does not strip leading whitespace from the text.
    This allows simpler text formatting of the script file using a constant text box size.

    This code has been commented out twice below:
    # while lines and not lines[0]:
    #     del lines[0]
    """

    def format_block(self, block):
        """Format the given block of text.

        This function is trimming leading and trailing
        empty lines and any leading whitespace that is common to all lines.

        Parameters
        ----------
        block : str
            block of text to be formatted

        """

        # Separate block into lines
        lines = str(block).splitlines()

        # Remove leading/trailing empty lines
        # while lines and not lines[0]:
        #     del lines[0]
        while lines and not lines[-1]:
            del lines[-1]

        # Look at first line to see how much indentation to trim
        try:
            ws = re.match(r"\s*", lines[0]).group(0)
        except Exception:
            ws = None
        if ws:
            lines = [x.replace(ws, "", 1) for x in lines]

        # Remove leading/trailing blank lines (after leading ws removal)
        # We do this again in case there were pure-whitespace lines
        # while lines and not lines[0]:
        #     del lines[0]
        while lines and not lines[-1]:
            del lines[-1]
        return "\n".join(lines) + "\n"

    # End of code taken from the word-wrapped text display module
