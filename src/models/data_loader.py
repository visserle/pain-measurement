import numpy as np
import polars as pl


def create_time_series_features(
    df: pl.DataFrame,
    columns: list[str],
    group_by_col: str = "sample_id",
) -> np.ndarray:
    """
    Convert polars DataFrame to numpy array formatted for time series modeling.

    Args:
        df: Input DataFrame
        columns: List of column names to use as features
        group_by_col: Column name to group by (default: "sample_id")

    Returns:
        numpy array with shape (samples, time steps, features)

    Note:
        Can be easily plotted using ipywidgets with the following code:
        ```python
        import ipywidgets


        @ipywidgets.interact(trial=(0, X.shape[0] - 1))
        def plot_trial(trial):
            for i in range(X.shape[2]):
                plt.plot(X[trial, :, i])
            plt.ylim(0, 1)
        ```
    """
    # Group by sample_id and aggregate all specified columns
    X = df.group_by(group_by_col, maintain_order=True).agg(
        [pl.col(col) for col in columns]
    )

    # Handle both univariate and multivariate cases
    if len(columns) == 1:
        X = np.vstack(X.get_column(columns[0]).to_numpy())
        X = np.expand_dims(X, axis=2)
    else:
        feature_arrays = [np.vstack(X.get_column(col).to_numpy()) for col in columns]
        X = np.stack(feature_arrays, axis=2)

    return X
