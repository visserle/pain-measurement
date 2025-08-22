import json
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from polars import col

from plots.make_model_plots import FIGURE_DIR
from src.data.database_manager import DatabaseManager

load_dotenv()
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))


def main():
    db = DatabaseManager()

    with db:
        # Get all questionnaire tables
        questionnaire_tables = {
            "BDI_II": "Questionnaire_BDI_II",
            "GENERAL": "Questionnaire_GENERAL",
            "MAAS": "Questionnaire_MAAS",
            "PANAS": "Questionnaire_PANAS",
            "PCS": "Questionnaire_PCS",
            "PHQ_15": "Questionnaire_PHQ_15",
            "PVAQ": "Questionnaire_PVAQ",
            "STAI_T_10": "Questionnaire_STAI_T_10",
        }

        results = {}

        # Process each questionnaire
        for questionnaire_name, table_name in questionnaire_tables.items():
            print(f"Processing {questionnaire_name}...")

            df = db.get_table(table_name)

            # Filter out participants with ID greater than 42
            df = df.filter(col("participant_id") < 43)

            # Remove columns starting with 'q' (q1, q2, etc.)
            q_columns = [
                col_name for col_name in df.columns if col_name.startswith("q")
            ]
            df = df.drop(q_columns)

            # Handle PANAS pre/post structure
            if questionnaire_name == "PANAS":
                # Add row index and create time column
                df = df.with_row_index()
                df = df.with_columns(
                    pl.when(col("index") % 2 == 0)
                    .then(pl.lit("pre"))
                    .otherwise(pl.lit("post"))
                    .alias("time")
                ).drop("index")

                # Pivot to have pre/post as separate columns
                df_pivot = df.pivot(
                    values=["positive_affect", "negative_affect"],
                    index=["participant_id", "age", "gender"],
                    on="time",
                )

                # Rename columns to be more descriptive
                df_pivot = df_pivot.rename(
                    {
                        "positive_affect_pre": "panas_positive_affect_pre",
                        "positive_affect_post": "panas_positive_affect_post",
                        "negative_affect_pre": "panas_negative_affect_pre",
                        "negative_affect_post": "panas_negative_affect_post",
                    }
                )

                results[questionnaire_name] = df_pivot
            else:
                # For other questionnaires, just add prefix to distinguish them
                # Keep only non-q columns and rename with questionnaire prefix
                columns_to_keep = ["participant_id", "age", "gender"]

                # Add questionnaire-specific columns with prefix
                for col_name in df.columns:
                    if col_name not in columns_to_keep and col_name != "questionnaire":
                        new_name = f"{questionnaire_name.lower()}_{col_name}"
                        df = df.rename({col_name: new_name})

                results[questionnaire_name] = df

        # Start with participant info from GENERAL questionnaire as base
        final_df = (
            results["GENERAL"]
            .with_columns(
                [
                    # Cast to string first, then handle the conversion
                    pl.col("general_weight")
                    .cast(pl.Utf8)
                    .str.replace(",", ".")
                    .cast(pl.Float64)
                    .alias("general_weight"),
                    pl.col("general_height")
                    .cast(pl.Utf8)
                    .str.replace(",", ".")
                    .cast(pl.Float64)
                    .alias("general_height"),
                ]
            )
            .with_columns(
                # Calculate BMI: weight (kg) / height (m)² - height is already in meters
                (pl.col("general_weight") / (pl.col("general_height")) ** 2)
                .alias("general_bmi")
                .round(0),
            )
            .select(
                [
                    "participant_id",
                    "age",
                    "gender",
                    "general_bmi",  # Include BMI instead of height and weight
                    "general_handedness",
                    "general_education",
                    "general_employment_status",
                    "general_physical_activity",
                    "general_meditation",
                    "general_contact_lenses",
                    "general_ear_wiggling",
                    "general_regular_medication",
                    "general_pain_medication_last_24h",
                ]
            )
            .with_columns(
                pl.when(col("gender") == "male")
                .then(pl.lit("m"))
                .otherwise(pl.lit("f"))
                .alias("gender")
            )
        )

        # Join with other questionnaires
        for questionnaire_name, df in results.items():
            if questionnaire_name != "GENERAL":
                # Select only the columns we want to join (excluding duplicate age, gender)
                join_columns = ["participant_id"]
                select_columns = ["participant_id"] + [
                    col_name
                    for col_name in df.columns
                    if col_name
                    not in ["participant_id", "age", "gender", "questionnaire"]
                ]

                df_to_join = df.select(select_columns)
                final_df = final_df.join(df_to_join, on="participant_id", how="left")

        # Convert to JSON
        print(f"\nFinal table shape: {final_df.shape}")
        print(f"Columns: {list(final_df.columns)}")

        # Convert to JSON-serializable format
        json_data = []
        for row in final_df.iter_rows(named=True):
            # Convert any potential numpy types to native Python types
            clean_row = {}
            for key, value in row.items():
                if value is None:
                    clean_row[key] = None
                elif isinstance(value, (int, float, str, bool)):
                    clean_row[key] = value
                else:
                    clean_row[key] = str(value)
            json_data.append(clean_row)

        # Save to JSON file
        with open(FIGURE_DIR / "questionnaire_results.json", "w") as f:
            json.dump(json_data, f, indent=2)

        print(
            f"\nSaved {len(json_data)} participant records to questionnaire_results.json"
        )

        # Show sample of the data
        print("\nSample data (first 2 participants):")
        print(final_df.head(2))

        return final_df


if __name__ == "__main__":
    result_df = main()
