SCORING_SCHEMAS = {
    "bdi-ii": {
        "components": {
            "total": range(1, 22),
        },
        "metric": "sum",
        "alert_threshold": 14,
        "alert_message": "depression",
    },
    "maas": {
        "components": {
            "total": range(1, 16),
        },
        "metric": "mean",
    },
    "panas": {
        "components": {
            "positive_affect": [1, 3, 4, 6, 10, 11, 13, 15, 17, 18],
            "negative_affect": [2, 5, 7, 8, 9, 12, 14, 16, 19, 20],
        },
        "metric": "mean",
    },
    "pcs": {
        "components": {
            "total": range(1, 14),
            "rumination": [8, 9, 10, 11],
            "magnification": [6, 7, 13],
            "helplessness": [1, 2, 3, 4, 5, 12],
        },
        "metric": "sum",
        "alert_threshold": 30,
        "alert_message": "a clinically significant level of catastrophizing",
    },
    "phq-15": {
        "components": {
            "total": range(1, 16),
        },
        "metric": "sum",
        "alert_threshold": 10,
        "alert_message": "a clinically significant level of somatic symptoms",
    },
    "pvaq": {
        "components": {
            "total": range(1, 17),
            "attention_to_pain": [1, 6, 7, 8, 10, 12, 13, 14, 15, 16],
            "attention_to_changes": [2, 3, 4, 5, 9, 11],
        },
        "reverse_items": [8, 16],
        "min_item_score": 0,
        "max_item_score": 5,
        "metric": "sum",
    },
    "stai-t-10": {
        "components": {
            "total": range(1, 11),
        },
        "reverse_items": [3, 4, 7],
        "min_item_score": 1,
        "max_item_score": 8,
        "metric": "percentage",
    },
}
