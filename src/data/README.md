Data transformation pipeline is `imotions` → `raw` → `interim` → `merged`, where

- `imotions`: data dump of all raw data for the whole experiment
- `raw`: relevant raw data of each trial
- `interim`: cleaned raw data for each participant
- `merged`: preprocessed data of all participants in one csv

The available features in each step are defined in the data config files.