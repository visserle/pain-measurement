import toml
import yaml
from expyriment import stimuli


def load_configuration(file_path):
    """Load configuration from a TOML file."""
    with open(file_path, "r", encoding="utf8") as file:
        return toml.load(file)


def load_script(file_path):
    """Load script from a YAML file."""
    with open(file_path, "r", encoding="utf8") as file:
        return yaml.safe_load(file)


def prepare_stimuli(script):
    """Convert script strings to TextBox stimuli and preload them."""
    for key, value in script.items():
        script[key] = stimuli.TextBox(text=value, size=[600, 500], position=[0, -100], text_size=20)
        script[key].preload()


def warn_signal():
    """Play a warn signal."""
    stimuli.Tone(duration=500, frequency=440).play()
