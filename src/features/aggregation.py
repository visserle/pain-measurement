import polars as pl

from src.features.transformations import map_trials


@map_trials
def calculate_corr(
    df: pl.DataFrame,
    feature1: str,
    feature2: str,
) -> pl.DataFrame:
    """
    Calculate the correlation between two features (columns of a DataFrame) for each
    trial.
    """
    return (
        df.select([feature1, feature2])
        .corr()
        .gather_every(2)  # corr method returns a 2x2 table
        .rename({feature1: "Trial", feature2: "Correlation"})  # reporpuse the columns
        .with_columns(pl.Series("Trial", [df["Trial"][0]]))  # add the trial number
        .with_columns(pl.col("Correlation").round(2))
    )
