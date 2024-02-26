import logging

import toml
import yaml

from src.expyriment.custom_text_box import CustomTextBox

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


BASE_SCREEN_SIZE = (1920, 1200)


def load_configuration(file_path):
    """Load configuration from a TOML file."""
    with open(file_path, "r", encoding="utf8") as file:
        return toml.load(file)


def load_script(file_path):
    """Load script from a YAML file."""
    with open(file_path, "r", encoding="utf8") as file:
        return yaml.safe_load(file)


def prepare_script(script, text_size, text_box_size, parent_key=None):
    """
    Recursively convert script strings to CustomTextBox stimuli and preload them.

    (With special preloading for 'instruction' as they are shown with the visual analogue scale composition)
    """
    for key, value in script.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries, passing down the current key as the parent_key
            prepare_script(value, text_size, text_box_size, key)
        else:
            # Convert strings to CustomTextBox stimuli
            script[key] = CustomTextBox(
                text=value, size=text_box_size, position=[0, 0], text_size=text_size
            )
            # Preload the stimuli
            if parent_key != "instruction":
                script[key].preload()
            else:
                script[key].preload(inhibit_ogl_compress=True)


def _scale_ratio(screen_size, base_screen_size=BASE_SCREEN_SIZE) -> float:
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
    base_value: tuple[int, int], screen_size, base_screen_size=BASE_SCREEN_SIZE
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
