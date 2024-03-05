import logging
import re

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def extract_number(s):
    """Used to get the score from items with alternative options (e.g. 1a, 1b)."""
    match = re.search(r"\d+", s)
    return int(match.group()) if match else None


def score_bdi(answers):
    score = {}
    score["total"] = sum(extract_number(answers[f"q{qid}"]) for qid in range(1, 22))
    logging.info(f"BDI-II score: {score['total']}")
    if score["total"] >= 14:
        logging.warning("BDI-II score indicates depression.")
    return score


def score_pcs(answers):
    score = {}
    score["rumination"] = sum(
        extract_number(answers[f"q{qid}"]) for qid in [8, 9, 10, 11]
    )
    score["magnification"] = sum(
        extract_number(answers[f"q{qid}"]) for qid in [6, 7, 13]
    )
    score["helplessness"] = sum(
        extract_number(answers[f"q{qid}"]) for qid in [1, 2, 3, 4, 5, 12]
    )
    score["total"] = (
        score["rumination"] + score["magnification"] + score["helplessness"]
    )
    logger.info(f"PCS score: {score['total']}")
    if score["total"] >= 30:
        logger.warning(
            "PCS score indicates a clinically significant level of catastrophizing."
        )
    return score
