SAMPLE_DURATION_MS = 6000
RANDOM_SEED = 1337  # https://xcancel.com/karpathy/status/1249876052047884288
BATCH_SIZE = 64
N_EPOCHS = 100
N_TRIALS = 20
INTERVALS = {  # see stimulus generation notebook for details
    "increases": "strictly_increasing_intervals",
    "plateaus": "plateau_intervals",
    "decreases": "major_decreasing_intervals",
}
LABEL_MAPPING = {  # as of yet, only binary classification
    "increases": 0,
    "plateaus": 0,
    "decreases": 1,
}
OFFSET_MS = {  # offsets for the intervals in milliseconds
    "increases": 0,
    "plateaus": 5000,
    "decreases": 1000,
}
