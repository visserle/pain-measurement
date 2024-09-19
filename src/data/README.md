TODO: improve
# Data Pipeline

The database creation takes place in `database_manager.py`.
The data processing steps are: `Raw` → `Preprocess` → `Feature`.

- `Raw`: Raw data as collected from each trial of the experiment.
- `Preprocess`: Preprocessed data with cleaned and transformed columns.
- `Feature`: Extracted features from the preprocessed data for analysis and modeling.

Invalid participants and trials can be removed at any stage of the pipeline using the `remove_invalid` key in the `get_table` method.

Note that iMotions' output is loaded into the database as `raw` data.

---

- `database_schema.py` defines the schema for the database.
- `database_manager.py` is for insertig data into the database and extracting data from the database.
- `data_processing.py` coordinates the data processing by creating dataframes that are ready for insertion into the database as tables. For 'feature' tables (Stimulus, EEG, EDA, PPG, pupillometry, facial expressions) it uses the functions from the feature module (feature modules are decoupled from the database and can be used independently).