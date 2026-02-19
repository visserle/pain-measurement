# Features Module

This module provides a decoupled framework for physiological signal processing and feature engineering, built on Top of [Polars](https://pola.rs/). It is designed to be used independently on DataFrames or as part of the main data pipeline (`src/data/main.py`).

> [!NOTE]
> Some operations are datatype-specific; for example, `float64` is preferred for most time-series calculations, while certain metadata columns are handled as `Int64`.

## Dual-Pipeline Architecture

The module distinguishes between two fundamental processing approaches:

- **Causal (Feature)**: Designed for classification tasks. These functions (e.g., `eda.py`) use causal filters and transformations, making them suitable for real-time applications where future data is unavailable.
- **Non-Causal (Explore)**: Designed for exploratory analysis and visualization. These functions (prefixed with `explore_`, e.g., `explore_eda.py`) use non-causal transformations (like zero-phase filtering or detrending) that utilize the entire signal history to produce cleaner results for analysis.

## Supported Modalities

The module includes specialized processing for various physiological and experimental data:

- **EEG** (`eeg.py`): Scalp electrification preprocessing and feature extraction.
- **EDA** (`eda.py`, `explore_eda.py`): Electrodermal activity measurement, including phasic and tonic component decomposition.
- **Heart Rate** (`hr.py`): PPG-based heart rate and heart rate variability features.
- **Pupillometry** (`pupil.py`, `explore_pupil.py`): Eye tracking and pupil diameter analysis.
- **Facial Expressions** (`face.py`): Analysis of facial action units.
- **Stimulus** (`stimulus.py`): Handling of experimental stimulus parameters and temperatures.

## Core Utilities

Processing is supported by a set of shared utilities:

- **Resampling** (`resampling.py`): Sophisticated downsampling and interpolation using `scipy.signal.decimate`. Note that timestamps are handled as floats (not `pl.Duration`) for broad compatibility.
- **Transformations** (`transforming.py`): Decorator-based system for applying functions across consistent groups:
    - `@map_participants`: Parallel processing across individual subjects.
    - `@map_trials`: Applying transformations per experimental trial.
- **Filtering** (`filtering.py`, `explore_filtering.py`): Standardized causal and non-causal Butterworth filters.
- **Labels** (`labels.py`): Functions for assigning stimulus-based categorical and continuous labels to time-series data.

## Usage Example

Feature modules are designed to work directly on Polars DataFrames:

```python
import polars as pl
from src.features.eda import feature_eda

df = pl.read_csv("raw_data.csv")
transformed_df = feature_eda(df)
```