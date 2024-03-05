from datetime import datetime
from pathlib import Path

import yaml
from flask import Flask, redirect, render_template, request, url_for

from src.log_config import configure_logging
from src.questionnaires.save_results import save_bdi, save_pcs
from src.questionnaires.score_results import score_bdi, score_pcs

configure_logging()

scale = "pcs"

app = Flask(__name__)

with open(f"src/questionnaires/inventory/{scale}.yaml", "r") as file:
    questionnaire = yaml.safe_load(file)


@app.route("/", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        answers = request.form
        score = score_pcs(answers)
        save_pcs(questionnaire, answers, score)
        return redirect(url_for("thank_you"))
    return render_template(
        f"{scale}.html",
        questions=questionnaire["questions"],
        options=questionnaire["options"] if "options" in questionnaire else None,
        title=questionnaire["title"] if "title" in questionnaire else None,
        instructions=questionnaire["instructions"]
        if "instructions" in questionnaire
        else None,
    )


@app.route("/thank_you")
def thank_you():
    return "Thank you for completing the survey!"


if __name__ == "__main__":
    app.run(debug=True)
