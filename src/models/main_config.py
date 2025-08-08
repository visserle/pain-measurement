SAMPLE_DURATION_MS = 7000
RANDOM_SEED = 1337  # https://x.com/fchollet/status/1612555896491749376
BATCH_SIZE = 64
N_EPOCHS = 100
N_TRIALS = 20
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
