import polars as pl

from src.features.transformations import map_trials

ADDITIONAL_INFO_COLUMNS = ["Trial", "Participant", "Stimulus_Seed", "Skin_Area"]


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
        .rename({feature1: "Trial", feature2: "Correlation"})  # repurpose the columns
        .with_columns(  # add additional info columns using list comprehension
            [
                pl.Series(add_column, [df[add_column][0]])
                for add_column in ADDITIONAL_INFO_COLUMNS
                if add_column in df.columns
            ]
        )
    )
