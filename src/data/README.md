# Data Pipeline

The data pipeline consists of three steps (`Raw` → `Preprocess` → `Feature`) over all modalities (EEG, EDA, PPG, pupillometry, facial expressions, and rating / temperatures), with each step corresponding to a table in the duckDB database. In a final step, the feature data is merged into a single table and labeled according to the stimulus function.

- `Raw`: Raw data from iMotions .csv output (without inter-trial data).
- `Preprocess`: Preprocessed data with cleaned and transformed columns.
- `Feature`: Extracted features from the preprocessed data.
- `Merged_and_Labeled_Data`: Merged feature data with stimulus labels, resampled to 10 Hz at equidistant time points.

Furthermore, there are additional tables for the experiment metadata, calibration results, and questionnaire responses.

## Files

- `database_schema.py` defines the schema for the database.
- `database_manager.py` is for insertig data into the database and extracting data from the database.
- `data_processing.py` coordinates the data processing by creating dataframes that are ready for insertion into the database as tables. For time series data tables (Stimulus, EEG, EDA, PPG, pupillometry, facial expressions) it uses the functions from the feature module (feature modules are decoupled from the database and can be used independently on dataframes).
- `data_config.py` contains very basic configuration parameters for the data pipeline and paths to the different data sources.

## Inavlid Data Handling

- Invalid participants were excluded from the analysis from the very beginning. Their data is not present in the database.
- Invalid trials with thermode or rating issues were exclude from all measurement tables, from the trial tables and from the PANAS questionnaire as the pre post values are not valid.
- Trials with measurement issues of one or more modalities can be excluded using the `exclude_trials_with_measurement_problems` keyword from the `get_table` method.
