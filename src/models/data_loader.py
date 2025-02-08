import logging

import numpy as np
import polars as pl
import torch
from polars import col
from torch.utils.data import DataLoader, TensorDataset


def transform_sample_df_to_arrays(
    sample_df: pl.DataFrame,
    feature_columns: list[str],
    label_column: str | None = "label",
    group_column: str | None = "participant_id",
    group_by_col: str = "sample_id",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Convert polars DataFrame to numpy arrays formatted for time series modeling.

    Args:
        sample_df: Input DataFrame
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
        X can be easily plotted using ipywidgets:
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
    X = sample_df.group_by(group_by_col, maintain_order=True).agg(
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
            sample_df.group_by(group_by_col)
            .agg(col(label_column).first())
            .get_column(label_column)
            .to_numpy()
        )

    # Get groups if specified
    groups = None
    if group_column:
        groups = (
            sample_df.group_by(group_by_col)
            .agg(col(group_column).first())
            .get_column(group_column)
            .to_numpy()
        )

    return X, y, groups


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    is_test: bool = True,
) -> tuple[DataLoader, DataLoader]:
    # Sanity check
    assert len(X_train.shape) == len(X_test.shape) == 3, (
        "X_train and X_test must have 3 dimensions: (samples, timesteps, features)"
    )

    train_data = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).view(-1, 1),
    )
    test_data = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).view(-1, 1),
    )
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    dataset = "Test" if is_test else "Validation"
    logging.debug(
        f"Train Data: {len(train_data)} samples, "
        f"{dataset} Data: {len(test_data)} samples"
    )

    return train_loader, test_loader
