import argparse
import webbrowser

import yaml
from flask import Flask, redirect, render_template, request, url_for

from src.log_config import configure_logging
from src.questionnaires.evaluation import save_results, score_results

configure_logging(ignore_libs=["werkzeug", "participant_data"])

app = Flask(__name__)


def load_questionnaire(scale):
    with open(f"src/questionnaires/inventory/{scale}.yaml", "r") as file:
        current_questionnaire = yaml.safe_load(file)
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
            return redirect(url_for("thank_you"))

    return render_template(
        f"{scale}.html.j2",
        title=current_questionnaire["title"],
        instructions=current_questionnaire["instructions"]
        if "instructions" in current_questionnaire
        else None,
        questions=current_questionnaire["questions"],
        options=current_questionnaire["options"]
        if "options" in current_questionnaire
        else None,
    )


@app.route("/thank_you")
def thank_you():
    return "Vielen Dank für das Ausfüllen der Fragebögen!"


if __name__ == "__main__":
    QUESTIONNAIRES = ["bdi", "maia", "pcs"]

    parser = argparse.ArgumentParser(
        description="Run the app with selected questionnaires."
    )
    parser.add_argument(
        "questionnaire",
        nargs="*",
        default=QUESTIONNAIRES,
        help=f"Select the questionnaires to run ({', '.join(QUESTIONNAIRES)}).",
    )
    args = parser.parse_args()

    global questionnaires
    questionnaires = args.questionnaire

    webbrowser.open_new("http://localhost:5000")
    app.run(debug=False)
