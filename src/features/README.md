# Features Module

This module provides a decoupled framework for physiological signal processing and feature engineering, built on Top of Polars. It is designed to be used independently on DataFrames or as part of the main data pipeline (`src/data/main.py`).

> [!NOTE]
> Some operations are datatype-specific; for example, `float64` is preferred for most time-series calculations, while certain metadata columns are handled as `Int64`.

## Dual-Pipeline Architecture

The module distinguishes between two fundamental processing approaches:

- **Causal (Feature)**: Designed for classification tasks. These functions (e.g., `eda.py`) use causal filters and transformations, making them suitable for real-time applications where future data is unavailable.
- **Non-Causal (Explore)**: Designed for exploratory analysis and visualization. These functions (prefixed with `explore_`, e.g., `explore_eda.py`) use non-causal transformations (like zero-phase filtering or detrending) that utilize the entire signal history to produce cleaner results for analysis.

