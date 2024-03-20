import argparse
import logging
import os
import signal
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import markdown
import requests
import yaml
from flask import Flask, after_this_request, redirect, render_template, request, url_for

from src.experiments.participant_data import (
    read_last_participant,
)
from src.experiments.questionnaires.evaluation import (
    save_results,
    score_results,
)
from src.log_config import configure_logging

LOG_DIR = Path("runs/experiments/questionnaires/")
INVENTORY_DIR = Path("src/experiments/questionnaires/inventory/")
QUESTIONNAIRES = [
    "general",  # without scoring
    "panas",
    "maia-2",
    "pcs",
    "pvaq",
    "lot-r",
    "bdi-ii",
    "brs",
    "stai-t-10",
    "erq",
    "maas",
]

parser = argparse.ArgumentParser(
    description="Run the app with selected questionnaires."
)
parser.add_argument(
    "questionnaire",
    nargs="*",
    default=QUESTIONNAIRES,
    help=f"Select the questionnaires to run: {', '.join(QUESTIONNAIRES)}.",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode.",
)

args = parser.parse_args()
questionnaires = args.questionnaire

app = Flask(__name__)

# Configure logging
log_file = LOG_DIR / datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S.log")
configure_logging(
    stream_level=logging.INFO if not args.debug else logging.DEBUG,
    file_path=log_file if not args.debug else None,
    ignore_libs=["werkzeug", "urllib3", "requests"],
)

# Load participant data
participant_info = read_last_participant()
if args.debug:
    logging.warning("Debug mode is enabled. Participant data will not be saved.")


def load_questionnaire(scale):
    with open(f"{INVENTORY_DIR / scale}.yaml", "r", encoding="utf-8") as file:
        current_questionnaire = yaml.safe_load(file)

    if "instructions" in current_questionnaire:
        current_questionnaire["instructions"] = markdown.markdown(
            current_questionnaire["instructions"]
        )
    return current_questionnaire


@app.route("/")
def home():
    return redirect(url_for("questionnaire_handler", scale=questionnaires[0]))


@app.route("/<scale>", methods=["GET", "POST"])
def questionnaire_handler(scale):
    current_questionnaire = load_questionnaire(scale)
    if request.method == "POST":
        answers = request.form
        logging.debug(f"Received answers: {answers}") if args.debug else None
        score = score_results(scale, answers) if scale != "general" else None
        save_results(
            participant_info,
            scale,
            current_questionnaire,
            answers,
            score,
        ) if not args.debug else None

        next_index = questionnaires.index(scale) + 1
        if next_index < len(questionnaires):
            return redirect(
                url_for("questionnaire_handler", scale=questionnaires[next_index])
            )
        else:
            return redirect(url_for("thanks"))

    return render_template(
        f"{current_questionnaire['layout']}.html.j2",
        title=current_questionnaire["title"],
        instructions=current_questionnaire["instructions"]
        if "instructions" in current_questionnaire
        else None,
        spectrum=current_questionnaire["spectrum"]
        if "spectrum" in current_questionnaire
        else None,
        options=current_questionnaire["options"]
        if "options" in current_questionnaire
        else None,
        questions=current_questionnaire["questions"],
    )


@app.route("/thanks")
def thanks():
    logging.info("Completed all questionnaires.")
    trigger_shutdown()

    return render_template(
        "thanks.html.j2",
        text="Vielen Dank für das Ausfüllen der Fragebögen!",
    )


@app.route("/shutdown")
def shutdown():
    os.kill(os.getpid(), signal.SIGINT)
    return "Server shutting down..."


def trigger_shutdown():
    @after_this_request
    def shutdown(response):
        threading.Timer(
            0.5, lambda: requests.get("http://localhost:5000/shutdown")
        ).start()
        return response

    return ""


def main():
    logging.info(
        f"Running questionnaire app with the following questionnaires: {list(map(str.upper, questionnaires))}"
    )
    webbrowser.open_new("http://localhost:5000")
    app.run(debug=args.debug)
    # NOTE: possible to deploy to web via docker: https://www.youtube.com/watch?v=cw34KMPSt4k


if __name__ == "__main__":
    main()
