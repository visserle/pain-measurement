import numpy as np
import polars as pl
from polars import col

from src.data.database_manager import DatabaseManager
from src.features.transforming import merge_dfs


def transform_df_to_arrays(
    df: pl.DataFrame,
    feature_columns: list[str],
    label_column: str | None = "label",
    group_column: str | None = "participant_id",
    group_by_col: str = "sample_id",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert polars DataFrame to numpy arrays formatted for time series modeling.

    Args:
        df: Input DataFrame
        feature_columns: List of column names to use as features
        label_column: Column name for labels (optional)
        group_column: Column name for groups (optional)
        group_by_col: Column name to group by (default: "sample_id")

    Returns:
        tuple of:
        - X: numpy array with shape (samples, time steps, features)
        - y: numpy array of labels (if label_column provided)
        - groups: numpy array of groups (if group_column provided)

    Note:
        X can be easily plotted using ipywidgets with the following code:
        ```python
        import ipywidgets


        @ipywidgets.interact(trial=(0, X.shape[0] - 1))
        def plot_trial(trial):
            for i in range(X.shape[2]):
                plt.plot(X[trial, :, i])
            plt.ylim(0, 1)
        ```
    """
    # Group by sample_id and aggregate features
    X = df.group_by(group_by_col, maintain_order=True).agg(
        [col(column) for column in feature_columns]
    )

    # Handle both univariate and multivariate cases
    if len(feature_columns) == 1:
        X = np.vstack(X.get_column(feature_columns[0]).to_numpy())
        X = np.expand_dims(X, axis=2)
    else:
        feature_arrays = [
            np.vstack(X.get_column(column).to_numpy()) for column in feature_columns
        ]
        X = np.stack(feature_arrays, axis=2)

    # Get labels if specified
    y = None
    if label_column:
        y = (
            df.group_by(group_by_col)
            .agg(col(label_column).first())
            .get_column(label_column)
            .to_numpy()
        )

    # Get groups if specified
    groups = None
    if group_column:
        groups = (
            df.group_by(group_by_col)
            .agg(col(group_column).first())
            .get_column(group_column)
            .to_numpy()
        )

    return X, y, groups
