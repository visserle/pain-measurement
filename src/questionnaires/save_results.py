import csv
import os
from datetime import datetime
from pathlib import Path

from src.expyriment.participant_data import read_last_participant

PARTICIPANT_DATA = Path("runs/expyriment/participants.xlsx")


def save_bdi(questionnaire, answers, score):
    fieldnames = [
        "timestamp",
        "id",
        "total_score",
    ] + [f'q{q["id"]}' for q in questionnaire["questions"]]
    file_exists = os.path.isfile("runs/questionnaires/bdi_results.csv")

    with open("runs/questionnaires/bdi_results.csv", mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            "timestamp": str(datetime.now())[0:19],
            "id": read_last_participant(PARTICIPANT_DATA)["id"],
            "total_score": score["total"],
        }
        row.update(
            {f'q{q["id"]}': answers[f'q{q["id"]}'] for q in questionnaire["questions"]}
        )
        writer.writerow(row)


def save_pcs(questionnaire, answers, score):
    fieldnames = [
        "timestamp",
        "id",
        "total_score",
        "rumination",
        "magnification",
        "helplessness",
    ] + [f"q{q['id']}" for q in questionnaire["questions"]]
    file_exists = os.path.isfile("runs/questionnaires/pcs_results.csv")

    with open("runs/questionnaires/pcs_results.csv", mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            "timestamp": str(datetime.now())[0:19],
            "id": read_last_participant(PARTICIPANT_DATA)["id"],
            "total_score": score["total"],
            "rumination": score["rumination"],
            "magnification": score["magnification"],
            "helplessness": score["helplessness"],
        }
        row.update(
            {f'q{q["id"]}': answers[f'q{q["id"]}'] for q in questionnaire["questions"]}
        )
        writer.writerow(row)
