Data transformations are based on the abstract `process_data.py` and the respective data config.

Two data transformations need special handling. This is taken into account in the abstract function via isinstance() checks of the config.

- `imotions → raw`: imotions files have inconsistent naming (up to now, TODO: export function)
- `raw → trial`: trial info has to be merged into each dataframe

All other data transformation rely on "atomic" functions, meaning that they only act on one specific dataframe of a signal.
