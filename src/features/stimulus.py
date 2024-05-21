import polars as pl

from src.features.transformations import map_trials


@map_trials
def calculate_corr(df: pl.DataFrame) -> pl.DataFrame:
    correlation_df = (
        df.select(["Rating", "Temperature"])
        .corr()
        .gather_every(2)  # corr method returns a 2x2 table
        .rename({"Rating": "Trial", "Temperature": "Correlation"})
        .with_columns(pl.Series("Trial", [df["Trial"][0]]))
        .with_columns(pl.col("Correlation").round(2))
    )
    return correlation_df
