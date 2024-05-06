"""Utility functions for expyriment experiments."""

import re
from pathlib import Path

import toml
import yaml
from expyriment.stimuli import Audio, TextBox
from google.cloud import texttospeech
from gtts import gTTS
from openai import OpenAI

BASE_SCREEN_SIZE = (1920, 1200)


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
    Calculate the adjusted value based on the screen size for 1D values like length,
    width or text size.

    Parameters:
    - base_value: int or float, base value to scale from
    - screen_size: tuple, current screen size (width, height)
    - base_screen_size: tuple, base screen size (width, height) for scaling reference,
      default=(1920, 1200)

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
    Calculate the adjusted value based on the screen size for 2D values like position or
    size.

    Parameters:
    - base_value: tuple, base value to scale from (width, height)
    - screen_size: tuple, current screen size (width, height)
    - base_screen_size: tuple, base screen size (width, height) for scaling reference,
      default=(1920, 1200)

    Returns:
    - scaled_value: tuple, scaled value based on the current screen size
    """
    scale_factor = _scale_ratio(screen_size, base_screen_size)
    scaled_value = (
        int(base_value[0] * scale_factor),
        int(base_value[1] * scale_factor),
    )
    return scaled_value


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
    Recursively convert existings script strings to CustomTextBox stimuli and preload
    them.

    (With special preloading for 'instruction' key as they are shown with the visual
    analogue scale composition)
    """
    for key, value in script.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries,
            # passing down the current key as the parent_key
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
            # Special preloading for 'instruction' which contains the VAS composition
            if parent_key == "instruction":
                script[key].preload(inhibit_ogl_compress=True)
            else:
                script[key].preload()


class CustomTextBox(TextBox):
    """
    Expyriment text box without leading whitespace stripping.

    This class is a copy of the TextBox class from expyriment.stimuli.textbox
    with the only difference that it does not strip leading whitespace from the text.
    This allows simpler text formatting of the script file using a constant text box
    size.

    This code has been commented out twice below:
    # while lines and not lines[0]:
    #     del lines[0]
    """

    def format_block(self, block) -> str:
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


def text_to_speech(
    text: str,
    output_path: str,
    model: str,
) -> None:
    """
    Generate and save an audio file from the given text.
    """
    # TODO: only keep one of the following implementations
    match model:
        case "openai":
            client = OpenAI()
            response = client.audio.speech.create(
                model="tts-1-hd",
                voice="alloy",
                input=text,
            )
            response.write_to_file(output_path)
        case "gtts":
            tts = gTTS(text, lang="de")
            tts.save(output_path)
        case "google":
            # Instantiates a client
            client = texttospeech.TextToSpeechClient()

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code="de-DE",
                name="de-DE-Wavenet-B",
            )

            # Select the type of audio file you want returned
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16
            )

            # Perform the text-to-speech request on the text input with the selected
            # voice parameters and audio file type
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )

            # The response's audio_content is binary.
            with open(output_path, "wb") as out:
                # Write the response to the output file.
                out.write(response.audio_content)
    print("Audio content written to file:", output_path)


def script_to_speech(
    script: dict,
    audio_dir: str,
    parent_key=None,
) -> None:
    """
    Generate audio files for all the script values.
    """
    for key, value in script.items():
        if isinstance(value, dict):
            # Recursively process nested dictionaries,
            # passing down the current key as the parent_key
            script_to_speech(
                script=value,
                audio_dir=audio_dir,
                parent_key=key,
            )
        else:
            # Remove '(y/n)' and '(Leertaste drücken, um fortzufahren)' from the text
            value = re.sub(r"\(y/n\)", "", value)
            value = re.sub(r"\(Leertaste drücken, um fortzufahren\)", "", value)

            audio_path = (
                Path(audio_dir) / f"{parent_key}_{key}.wav"
                if parent_key
                else Path(audio_dir) / f"{key}.wav"
            )
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            text_to_speech(value, audio_path, model="google")


def main_tts():
    """Generate audio files for the calibration and measurement scripts."""
    experiments = ["calibration", "measurement"]
    for exp in experiments:
        # Load the script file
        file_path = Path(f"src/experiments/{exp}/{exp}_script.yaml")
        script = load_script(file_path)
        # Create audio files for the script
        script_to_speech(script, f"src/experiments/{exp}/audio")


def prepare_audio(
    audio: dict,
    audio_dir: str,
    parent_key: str = None,
) -> None:
    """
    Recursively convert existings audio files to Audio stimuli and preload them,
    based on the structure of the script dictionary.
    """
    for key, value in audio.items():
        print(key)
        if isinstance(value, dict):
            # Recursively process nested dictionaries,
            # passing down the current key as the parent_key
            prepare_audio(value, audio_dir, parent_key=key)
        else:
            audio_path = (
                Path(audio_dir) / f"{parent_key}_{key}.wav"
                if parent_key
                else Path(audio_dir) / f"{key}.wav"
            ).as_posix()
            print(audio_path)
            audio[key] = Audio(audio_path)
            audio[key].preload()


if __name__ == "__main__":
    main_tts()
