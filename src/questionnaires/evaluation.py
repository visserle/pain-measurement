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
    "maia": {
        "components": {
            "noticing": [1, 2, 3, 4],
            "not_distracting": [5, 6, 7, 8, 9, 10],  # items are reverse-scored
            "not_worrying": [11, 12, 13, 14, 15],  # some items are reverse-scored
            "attention_regulation": [16, 17, 18, 19, 20, 21, 22],
            "emotional_awareness": [23, 24, 25, 26, 27],
            "self_regulation": [28, 29, 30, 31],
            "body_listening": [32, 33, 34],
            "trusting": [35, 36, 37],
        },
        "reverse_scored": [5, 6, 7, 8, 9, 10, 11, 12, 15],
        "max_item_score": 5,
        # No alert_threshold or alert_message provided in the document
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
        component_score = 0
        for qid in questions:
            item_score = _extract_number(answers.get(f"q{qid}"))
            # Reverse score if necessary
            if qid in schema.get("reverse_scored", []):
                item_score = schema["max_item_score"] - item_score
            component_score += item_score
        score[component] = component_score

    # Calculate total score
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
    if len(SCORING_SCHEMAS[scale]["components"]) > 1:
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
