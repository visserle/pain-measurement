import csv
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from src.expyriment.participant_data import read_last_participant
from src.questionnaires.scoring_schemas import SCORING_SCHEMAS

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

PARTICIPANT_DATA = Path("runs/expyriment/participants.xlsx")
RESULTS_DIRECTORY = Path("runs/questionnaires")


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
            if qid in schema.get("filler_items", []):
                continue
            item_score = _extract_number(answers.get(f"q{qid}"))
            if qid in schema.get("reverse_items", []):
                item_score = (schema["max_item_score"] - item_score) + schema[
                    "min_item_score"
                ]
            component_score += item_score
        score[component] = component_score

    # Recalculate scores based on special metric
    if schema.get("metric") == "mean":
        if "total" in score:
            score["total"] = round(
                score["total"] / len(schema["components"]["total"]), 2
            )
        else:  # if there is no "total" component, apply metric to all components
            for key, value in score.items():
                score[key] = round(value / len(schema["components"][key]), 2)
    elif schema.get("metric") == "percentage":
        # only used for STAI-T-10 on the total score
        min_score = schema["min_item_score"] * len(schema["components"]["total"])
        max_score = schema["max_item_score"] * len(schema["components"]["total"])
        score["total"] = round(
            (score["total"] - min_score) / (max_score - min_score) * 100, 2
        )

    formatted_score = ", ".join(f"{key}: {value}" for key, value in score.items())
    logger.info(f"{scale.upper()} score = {formatted_score}.")
    if "alert_threshold" in schema and score["total"] >= schema["alert_threshold"]:
        logger.warning(f"{scale.upper()} score indicates {schema['alert_message']}.")

    return score


def save_results(scale, questionnaire, answers, score):
    filename = RESULTS_DIRECTORY / f"{scale}_results.csv"
    file_exists = os.path.isfile(filename)

    # Basic fieldnames
    fieldnames = ["timestamp", "id", "age", "gender"]
    # Extend the fieldnames with scale-specific components and question IDs (raw answers)
    fieldnames.extend(SCORING_SCHEMAS[scale]["components"].keys())
    fieldnames.extend([f"q{q['id']}" for q in questionnaire["questions"]])

    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        # Construct and write the row directly in the CSV file
        participant_info = read_last_participant(PARTICIPANT_DATA)
        row = {
            "timestamp": str(datetime.now())[0:19],
            "id": participant_info["id"],
            "age": participant_info["age"],
            "gender": "M" if participant_info["gender"] == "Male" else "F",
        }
        row.update({component: score[component] for component in score})
        row.update(
            {f'q{q["id"]}': answers[f'q{q["id"]}'] for q in questionnaire["questions"]}
        )
        writer.writerow(row)
