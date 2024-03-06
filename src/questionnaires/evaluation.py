import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from src.expyriment.participant_data import read_last_participant

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

PARTICIPANT_DATA = Path("runs/expyriment/participants.xlsx")
RESULTS_DIRECTORY = Path("runs/questionnaires")

SCORING_SCHEMAS = {
    "bdi": {
        "components": {
            "total": range(1, 22),
        },
        "alert_threshold": 14,
        "alert_message": "depression",
    },
    "pcs": {
        "components": {
            "rumination": [8, 9, 10, 11],
            "magnification": [6, 7, 13],
            "helplessness": [1, 2, 3, 4, 5, 12],
        },
        "alert_threshold": 30,
        "alert_message": "a clinically significant level of catastrophizing",
    },
}


def _extract_number(s):
    """Used to get the score from items with alternative options (e.g. 1a, 1b)."""
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


def score_results(scale, answers):
    score = {}
    schema = SCORING_SCHEMAS[scale]

    for component, questions in schema["components"].items():
        score[component] = sum(_extract_number(answers[f"q{qid}"]) for qid in questions)

    score["total"] = sum(score.values())
    logger.info(f"{scale.upper()} score: {score['total']}")

    if "alert_threshold" in schema and score["total"] >= schema["alert_threshold"]:
        logger.warning(f"{scale.upper()} score indicates {schema['alert_message']}.")

    return score


def save_results(scale, questionnaire, answers, score):
    filename = RESULTS_DIRECTORY / f"{scale}_results.csv"
    file_exists = os.path.isfile(filename)

    # Basic fieldnames include timestamp, participant ID, and total score
    fieldnames = ["timestamp", "id", "total_score"]
    # Extend the fieldnames with scale-specific components and question IDs
    fieldnames.extend(SCORING_SCHEMAS[scale]["components"].keys())
    fieldnames.extend([f"q{q['id']}" for q in questionnaire["questions"]])

    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Construct and write the row directly in the CSV file
        row = {
            "timestamp": str(datetime.now())[0:19],
            "id": read_last_participant(PARTICIPANT_DATA)["id"],
            "total_score": score["total"],
        }
        row.update(
            {component: score[component] for component in score if component != "total"}
        )
        row.update(
            {f'q{q["id"]}': answers[f'q{q["id"]}'] for q in questionnaire["questions"]}
        )
        writer.writerow(row)
