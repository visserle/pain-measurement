# Data Pipeline

Processing steps: `Raw` → `Preprocess` → `Feature`.

- `Raw`: Raw data as collected from each trial of the experiment.
- `Preprocess`: Preprocessed data with cleaned and transformed columns.
- `Feature`: Extracted features from the preprocessed data for analysis and modeling.

Invalid participants and trials can be removed at any stage of the pipeline using the `remove_invalid` key in the `get_table` method.

Note that iMotions data is only available in the original .csv format and is not integrated into the database.
