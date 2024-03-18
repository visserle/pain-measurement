import logging
import re
from pathlib import Path

from src.expyriment.participant_data import add_participant_info
from src.questionnaires.scoring_schemas import SCORING_SCHEMAS

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])

RESULTS_DIR = Path("data/questionnaires")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _extract_number(string):
    """Used to get the score from items with alternative options (e.g. 1a, 1b)."""
    match = re.search(r"\d+", string)
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

    # Recalculate scores based on special metric (sum by default)
    if schema.get("metric") == "mean":
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


def save_results(participant_info, scale, questionnaire, answers, score):
    filename = RESULTS_DIR / f"{scale}_results.csv"

    # Basic fieldnames
    fieldnames = ["timestamp", "id", "age", "gender"]
    # Extend fieldnames with scale-specific components and question IDs (raw answers)
    if scale != "general":  # general questionnaire does not have scoring components
        fieldnames.extend(SCORING_SCHEMAS[scale]["components"].keys())
    fieldnames.extend([f"q{q['id']}" for q in questionnaire["questions"]])

    if scale != "general":
        participant_info.update({component: score[component] for component in score})
    participant_info.update(
        {f'q{q["id"]}': answers[f'q{q["id"]}'] for q in questionnaire["questions"]}
    )

    add_participant_info(filename, participant_info)
