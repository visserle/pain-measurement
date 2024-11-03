import logging
from functools import reduce, wraps

import polars as pl
from polars import col

logger = logging.getLogger(__name__.rsplit(".", maxsplit=1)[-1])


def map_by_group(group_col: str):
    """
    Generic decorator to apply a function to groups in a pl.DataFrame.

    Args:
        group_col (str): The column name to group by
    """

    def decorator(func: callable):
        @wraps(func)
        def wrapper(
            df: pl.DataFrame,
            *args,
            **kwargs,
        ) -> pl.DataFrame:
            try:
                df = df.group_by(group_col, maintain_order=True).map_groups(
                    lambda group: func(group, *args, **kwargs)
                )
            except pl.exceptions.ColumnNotFoundError:
                logger.warning(
                    f"No '{group_col}' column found. Applying function {func.__name__} "
                    "to the entire DataFrame."
                )
                df = func(df, *args, **kwargs)
            return df

        return wrapper

    return decorator


def map_trials(func: callable):
    """
    Decorator to apply a function to each trial in a pl.DataFrame.
    """
    return map_by_group("trial_id")(func)


def map_participants(func: callable):
    """
    Decorator to apply a function to each participant in a pl.DataFrame.
    """
    return map_by_group("participant_id")(func)


def merge_dfs(
    dfs: list[pl.DataFrame],
    on: list[str] = ["participant_id", "trial_id", "trial_number", "timestamp"],
    sort_by: list[str] = ["trial_id", "timestamp"],
) -> pl.DataFrame:
    """
    Merge multiple DataFrames into a single DataFrame.

    Default on and sort_by columns are set for data DataFrames (raw, preprocess,
    feature), however, these can be adjusted as needed for other DataFrames (e.g.,
    Trials).

    For merging e.g. Stimulus data with Trials information, use:
    ````
    df = merge_dfs(
        dfs=[stimulus, trials],
        on=["trial_id", "participant_id", "trial_number"],
    )
    ````
    """
    if len(dfs) < 2:
        raise ValueError(
            "A list with at least two DataFrames are required for merging."
        )

    df = reduce(
        lambda left, right: left.join(
            right,
            on=on,
            how="full",
            coalesce=True,
        )
        .sort(sort_by)
        .drop(["rownumber_right", "samplenumber_right"], strict=False),
        dfs,
    )
    return df
