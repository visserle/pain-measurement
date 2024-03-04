import csv
import logging
import os
import re
from datetime import datetime

import yaml
from flask import Flask, redirect, render_template, request, url_for

from src.log_config import configure_logging

configure_logging()


def extract_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


app = Flask(__name__)

with open("src/questionnaires/inventory/bdi-ii.yaml", "r") as file:
    questionnaire = yaml.safe_load(file)


def save_results(answers, total_score):
    fieldnames = ["timestamp", "total_score"] + [
        f'q{q["id"]}' for q in questionnaire["questions"]
    ]
    file_exists = os.path.isfile("survey_results.csv")

    with open("survey_results.csv", mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Only write header if the file does not exist

        row = {"timestamp": datetime.now(), "total_score": total_score}
        row.update(
            {f'q{q["id"]}': answers[f'q{q["id"]}'] for q in questionnaire["questions"]}
        )
        writer.writerow(row)


@app.route("/", methods=["GET", "POST"])
def survey():
    if request.method == "POST":
        answers = request.form
        total_score = sum(
            extract_number(answers[f'q{q["id"]}']) for q in questionnaire["questions"]
        )
        save_results(answers, total_score)
        return redirect(url_for("thank_you"))
    return render_template("survey.html", questions=questionnaire["questions"])


@app.route("/thank_you")
def thank_you():
    return "Thank you for completing the survey!"


if __name__ == "__main__":
    app.run(debug=True)
