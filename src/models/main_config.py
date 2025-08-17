RANDOM_SEED = 42
SAMPLE_DURATION_MS = 7000
BATCH_SIZE = 64
N_EPOCHS = 30
N_TRIALS = 30
INTERVALS = {  # see stimulus generation notebook for details
    "increases": "strictly_increasing_intervals",
    "plateaus": "plateau_intervals",
    "decreases": "major_decreasing_intervals",
}
LABEL_MAPPING = {  # binary classification
    "increases": 0,
    "plateaus": 0,
    "decreases": 1,
}
OFFSETS_MS = {  # offsets for the intervals in milliseconds
    "increases": 0,
    "plateaus": 5000,
    "decreases": 1000,
}
