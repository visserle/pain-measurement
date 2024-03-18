SCORING_SCHEMAS = {
    "bdi-ii": {
        "components": {
            "total": range(1, 22),
        },
        "metric": "sum",
        "alert_threshold": 14,
        "alert_message": "depression",
    },
    "brs": {
        "components": {
            "total": range(1, 7),
        },
        "reverse_items": [2, 4, 6],
        "min_item_score": 1,
        "max_item_score": 5,
        "metric": "mean",
    },
    "erq": {
        "components": {
            "reappraisal": [1, 3, 5, 7, 8, 10],
            "suppression": [2, 4, 6, 9],
        },
        "metric": "mean",
    },
    "lot-r": {
        "components": {
            # following two-dimensional model of indepentent optimism and pessimism
            "pessimism": [3, 7, 9],
            "optimism": [1, 4, 10],
        },
        "filler_items": [2, 5, 6, 8],
        "metric": "sum",
    },
    "maas": {
        "components": {
            "total": range(1, 16),
        },
        "metric": "mean",
    },
    "maia-2": {
        "components": {
            "noticing": [1, 2, 3, 4],
            "not_distracting": [5, 6, 7, 8, 9, 10],  # items are reverse-scored
            "not_worrying": [11, 12, 13, 14, 15],  # some items are reverse-scored
            "attention_regulation": [16, 17, 18, 19, 20, 21, 22],
            "emotional_awareness": [23, 24, 25, 26, 27],
            "self_regulation": [28, 29, 30, 31],
            "body_listening": [32, 33, 34],
            "trusting": [35, 36, 37],
        },
        "reverse_items": [5, 6, 7, 8, 9, 10, 11, 12, 15],
        "min_item_score": 0,
        "max_item_score": 5,
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
