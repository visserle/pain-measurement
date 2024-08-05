import logging
import re
from pathlib import Path

from src.experiments.participant_data import add_participant_info
from src.experiments.questionnaires.scoring_schemas import SCORING_SCHEMAS

RESULTS_DIR = Path("data/experiments/questionnaires")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def _extract_number(string: str) -> int:
    """Used to get the score from items with alternative options (e.g. 1a, 1b)."""
    match = re.search(r"\d+", string)
    return int(match.group()) if match else None


def score_results(
    scale: str,
    answers: dict,
) -> dict:
    """
    Calculate the score for each component of the questionnaire.
    """
    score = {}
    schema = SCORING_SCHEMAS.get(scale)

    if not schema:
        logger.error(
            f"No schema found for scale: {scale.upper()}. Returning empty score."
        )
        return score

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
        logger.error(f"{scale.upper()} score indicates {schema['alert_message']}.")

    return score


def save_results(
    participant_info: dict,
    scale: str,
    questionnaire: dict,
    answers: dict,
    score: dict,
) -> None:
    if not answers:
        logger.debug(
            "No answers available for participant: "
            f"{participant_info.get('id', 'unknown')}, scale: {scale.upper()}. "
            "Not saving results."
        )
        return

    filename = RESULTS_DIR / f"{scale}_results.csv"
    # Avoid modifying the original participant_info
    local_participant_info = participant_info.copy()

    # For general questionnaires we don't have a score and only save the answers
    if scale.split("_")[0] == "general":
        prefix = ""
    else:
        # Update participant info with scores
        local_participant_info |= {component: score[component] for component in score}
        prefix = "q"  # e.g. q1, q2, etc.

    # Add raw answers to the participant info
    local_participant_info |= {
        f'{prefix}{question["id"]}': answers[f'{prefix}{question["id"]}']
        for question in questionnaire.get("questions", [])
    }
    add_participant_info(local_participant_info, filename)
