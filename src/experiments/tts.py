"""Generate audio files for the calibration and measurement scripts using Google TTS."""

import re
from pathlib import Path

from google.cloud import texttospeech

from src.experiments.utils import load_script


def text_to_speech(
    text: str,
    output_path: str,
    model: str = "de-DE-Wavenet-B",
) -> None:
    """
    Generate and save an audio file from the given text.
    """
    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text.strip())

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code=model[:5],
        name=model,
    )

    # Select the type of audio file
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        speaking_rate=1.05,
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
            # Remove unnecessary text from the script
            value = re.sub(r"\(y/n\)", "", value)
            value = re.sub(r"\(Leertaste drücken, um fortzufahren\)", "", value)
            value = re.sub(r"Nun wechseln wir die Hautstelle am Arm.", "", value)

            audio_path = (
                Path(audio_dir) / f"{parent_key}_{key}.wav"
                if parent_key
                else Path(audio_dir) / f"{key}.wav"
            )
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            text_to_speech(value, audio_path)


def main_tts():
    """Generate audio files for the calibration and measurement scripts."""
    experiments = ["calibration", "measurement"]
    for exp in experiments:
        # Load the script file
        file_path = Path(f"src/experiments/{exp}/{exp}_script.yaml")
        script = load_script(file_path)
        # Create audio files for the script
        script_to_speech(script, f"src/experiments/{exp}/audio")


if __name__ == "__main__":
    main_tts()
