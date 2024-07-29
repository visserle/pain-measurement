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

QUESTIONNAIRES = [
    "general",  # no scoring
    "bdi-ii",
    "phq-15",
    "panas",
    "pcs",
    "pvaq",
    "stai-t-10",
    "maas",
]

INVENTORY_DIR = Path("src/experiments/questionnaires/inventory/")
LOG_DIR = Path("runs/experiments/questionnaires/")


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
    help="Enable debug mode. Data will not be saved.",
)
parser.add_argument(
    "-w",
    "--welcome",
    action="store_true",
    help="Show welcome page with instructions and information.",
)
args = parser.parse_args()
questionnaires = args.questionnaire

# Configure logging
log_file = LOG_DIR / datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S.log")
configure_logging(
    stream_level=logging.INFO if not args.debug else logging.DEBUG,
    file_path=log_file if not args.debug else None,
    ignore_libs=["werkzeug", "urllib3", "requests"],
)
if args.debug:
    logging.debug("Debug mode is enabled. Data will not be saved.")

# Load participant data
participant_info = read_last_participant() if not args.debug else dict(id=0)

app = Flask(__name__)


def load_questionnaire(scale: str) -> dict:
    try:
        with open(f"{INVENTORY_DIR / scale}.yaml", "r", encoding="utf-8") as file:
            current_questionnaire = yaml.safe_load(file)

        if "instructions" in current_questionnaire:
            current_questionnaire["instructions"] = markdown.markdown(
                current_questionnaire["instructions"]
            )
        return current_questionnaire
    except FileNotFoundError:
        logging.error(f"Questionnaire '{scale.upper()}' not found.")
        return dict(title=scale.upper())  # return empty questionnaire


@app.route("/", methods=["GET", "POST"])
def home():
    return (
        redirect("/welcome")  # hardcoded works all the time, url_for not reliable here
        if args.welcome
        else redirect(url_for("questionnaire_handler", scale=questionnaires[0]))
    )


@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")


@app.route("/welcome")
def welcome():
    args.welcome = False  # we only want to show the welcome page once

    with open(INVENTORY_DIR / "welcome.yaml", "r", encoding="utf-8") as file:
        welcome = yaml.safe_load(file)
        welcome["instructions"] = markdown.markdown(welcome["instructions"])

    return render_template(
        "welcome.html.j2", title=welcome["title"], instructions=welcome["instructions"]
    )


@app.route("/<scale>", methods=["GET", "POST"])
def questionnaire_handler(scale):
    current_questionnaire = load_questionnaire(scale)
    if request.method == "POST":
        answers = request.form
        logging.debug(f"{scale.upper()} answers = {answers}")
        score = (
            score_results(scale, answers) if scale.split("_")[0] != "general" else None
        )  # general questionnaire has no scoring
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
        f"{current_questionnaire['layout']}.html.j2"
        if "layout" in current_questionnaire
        else "blank.html.j2",
        title=current_questionnaire.get("title"),
        instructions=current_questionnaire.get("instructions"),
        spectrum=current_questionnaire.get("spectrum"),
        options=current_questionnaire.get("options"),
        questions=current_questionnaire.get("questions"),
    )


@app.route("/thanks")
def thanks():
    logging.info("Completed all questionnaires.")
    trigger_shutdown()  # shutdown the server after the response is sent

    return render_template(
        "thanks.html.j2",
        title="Vielen Dank für das Ausfüllen der Fragebögen!",
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
        "Running questionnaire app with the following questionnaires: "
        f"{list(map(str.upper, questionnaires))}"
    )
    webbrowser.open_new("http://localhost:5000")
    app.run(debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
