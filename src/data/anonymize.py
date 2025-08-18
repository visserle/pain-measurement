import logging

import numpy as np
import polars as pl

logger = logging.getLogger(__name__.rsplit(".", 1)[-1])


def anonymize_db(db):
    with db:
        # Remove invalid participants table
        db.execute("DROP TABLE Invalid_Participants")
        # Get invalid trials for mapping
        invalid_trials = db.get_table("invalid_trials")

    # Count trials per participant and filter for those with exactly 12 trials
    # (i.e. included participants with measurement probblems for one or more modalities)
    participants_with_problematic_trials_only = (
        invalid_trials.group_by("participant_id", maintain_order=True)
        .agg(pl.col("trial_number").n_unique().alias("trial_count"))
        .filter(pl.col("trial_count") == 12)
        .get_column("participant_id")
        .to_list()
    )

    with db:
        tables = db.execute("SHOW TABLES").fetchall()
        tables = list(map(lambda x: x[0], tables))
        tables.remove("Trials")
        tables.remove("Seeds")  # no participants here
        tables.insert(0, "Trials")  # 1st place

    Anonymizer = ID_Anonymizer(participants_with_problematic_trials_only)

    for table in tables:
        with db:
            df = db.get_table(table)
            df = Anonymizer.anonymize_participant_ids(df)
            db.ctas(table, df)
        logger.debug(f"Anonymized table: {table}")


class ID_Anonymizer:
    """
    Class to handle anonymization of participant IDs.

    The mapping of participant IDs is created only once and reused for subsequent calls.
    E.g. first use trials DataFrame to anonymize participant IDs, then use the same mapping to anonymize other DataFrame.
    """

    def __init__(
        self,
        participants_with_problematic_trials_only: list | None = None,
    ):
        self.participants_with_problematic_trials_only = (
            participants_with_problematic_trials_only or []
        )
        self._participant_mapping = None

    def anonymize_participant_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Anonymizes participant IDs in the df DataFrame.
        """
        sort_cols = ["participant_id"]
        if "trial_number" in df.columns:
            sort_cols.append("trial_number")
        if "timestamp" in df.columns:
            sort_cols.append("timestamp")

        participants = (
            df.get_column("participant_id").unique(maintain_order=True).to_numpy()
        )

        # Create mapping only once
        if self._participant_mapping is None:
            self._create_participant_mapping(participants)

        old_to_new = self._participant_mapping

        # Replace participant IDs in the DataFrame
        df = df.with_columns(
            pl.col("participant_id")
            .replace_strict(old_to_new, default=None)
            .alias("participant_id")
            .cast(pl.UInt8)
        ).sort(sort_cols, descending=False)

        # Reassign trial IDs to be sequential starting from 1
        if "trial_id" in df.columns:
            df = self._reassign_trial_ids(df)

        return df

    def _create_participant_mapping(self, participants):
        """Create mapping between original and anonymized participant IDs.

        We move problematic participants to the end of the list and assign them the highest IDs
        to improve plots."""
        sorted_participants = np.sort(participants)
        total_participants = len(sorted_participants)

        # Separate problematic and normal participants
        problematic = [
            p
            for p in sorted_participants
            if p in self.participants_with_problematic_trials_only
        ]
        normal = [
            p
            for p in sorted_participants
            if p not in self.participants_with_problematic_trials_only
        ]

        # Generate shuffled IDs for normal participants (1 to n-k)
        normal_ids = np.arange(1, len(normal) + 1, dtype=np.uint8)
        np.random.shuffle(normal_ids)

        # Assign highest IDs to problematic participants (n-k+1 to n)
        problematic_ids = np.arange(
            len(normal) + 1, total_participants + 1, dtype=np.uint8
        )

        # Create mapping
        participant_mapping = {}
        participant_mapping.update(dict(zip(normal, normal_ids)))
        participant_mapping.update(dict(zip(problematic, problematic_ids)))

        # Ensure the mapping is sorted by original participant IDs
        participant_mapping = dict(sorted(participant_mapping.items()))

        self._participant_mapping = participant_mapping

    def _reassign_trial_ids(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Reassigns trial IDs in the df DataFrame to be sequential starting from 1.
        """
        # Create mapping from your unique values
        unique_vals = df["trial_id"].unique(maintain_order=True).to_list()
        old_to_new = {val: idx + 1 for idx, val in enumerate(unique_vals)}

        return df.with_columns(
            pl.col("trial_id")
            .replace_strict(old_to_new, default=None)
            .alias("trial_id")
            .cast(pl.UInt16)
        )
