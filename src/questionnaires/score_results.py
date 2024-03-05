import logging
import re

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def extract_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


def score_bdi(questionnaire, answers):
    score = sum(
        extract_number(answers[f'q{q["id"]}']) for q in questionnaire["questions"]
    )
    logging.info(f"BDI-II score: {score}")
    if score >= 14:
        logging.warning("BDI-II score indicates depression.")
    return score
