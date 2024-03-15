# TODO
# add readme to inventory
# add full names of the scales, authors, and references
# add schema validation
# improve radio buttons
# add progress bar
# add fragebogen für allgemeines
# meditationserfahrungen abfragen, händigkeit, sehvermögen, hornhautverkrümmung, botox, demographiscshe daten (gerne doppelt), sozioökonomischer status, bildung, beruf, einkommen
# freitext für pcs an welchen schmerz gedacht wurde?

import argparse
import logging
import webbrowser
from datetime import datetime
from pathlib import Path

import markdown
import yaml
from flask import Flask, redirect, render_template, request, url_for

from src.log_config import configure_logging
from src.questionnaires.evaluation import save_results, score_results

# Configure logging
LOG_DIR = Path("runs/questionnaires/logs/")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / datetime.now().strftime(r"%Y_%m_%d__%H_%M_%S.log")
configure_logging(
    stream_level=logging.DEBUG,
    file_path=log_file,
    ignore_libs=["werkzeug", "participant_data"],
)

QUESTIONNAIRES = [
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

app = Flask(__name__)


def load_questionnaire(scale):
    with open(
        f"src/questionnaires/inventory/{scale}.yaml", "r", encoding="utf-8"
    ) as file:
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
        score = score_results(scale, answers)
        save_results(scale, current_questionnaire, answers, score)

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
        questions=current_questionnaire["questions"],
        options=current_questionnaire["options"]
        if "options" in current_questionnaire
        else None,
    )


@app.route("/thanks")
def thanks():
    logging.info("The questionnaires have been completed.")
    return render_template(
        "thanks.html.j2",
        text="Vielen Dank für das Ausfüllen der Fragebögen!",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the app with selected questionnaires."
    )
    parser.add_argument(
        "questionnaire",
        nargs="*",
        default=QUESTIONNAIRES,
        help=f"Select the questionnaires to run: {', '.join(QUESTIONNAIRES)}.",
    )
    args = parser.parse_args()

    global questionnaires
    questionnaires = args.questionnaire

    logging.info(f"Running the app with the following questionnaires: {questionnaires}")
    webbrowser.open_new("http://localhost:5000")
    app.run(debug=False)
