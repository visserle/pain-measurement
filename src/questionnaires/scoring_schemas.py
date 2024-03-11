SCORING_SCHEMAS = {
    "pcs": {
        "components": {
            "total": range(1, 14),
            "rumination": [8, 9, 10, 11],
            "magnification": [6, 7, 13],
            "helplessness": [1, 2, 3, 4, 5, 12],
        },
        "alert_threshold": 30,
        "alert_message": "a clinically significant level of catastrophizing",
    },
    "bdi-ii": {
        "components": {
            "total": range(1, 22),
        },
        "alert_threshold": 14,
        "alert_message": "depression",
    },
    "ffmq": {
        "components": {
            "nonjudging": [3, 10, 14, 17, 25, 30, 35, 39],
            "describing": [2, 7, 12, 16, 22, 27, 32, 37],
            "observing": [1, 6, 11, 15, 20, 26, 31, 36],
            "acting_with_awareness": [5, 8, 13, 18, 23, 28, 34, 38],
            "nonreactivity": [4, 9, 19, 21, 24, 29, 33],
        },
        "reverse_items": [
            3,
            10,
            14,
            17,
            25,
            30,
            35,
            39,
            12,
            16,
            22,
            5,
            8,
            13,
            18,
            23,
            28,
            34,
            38,
        ],
        "min_item_score": 1,
        "max_item_score": 6,
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
    "lot-r": {
        "components": {
            # following two-dimensional model of indepentent optimism and pessimism
            "pessimism": [3, 7, 9],
            "optimism": [1, 4, 10],
        },
        "filler_items": [2, 5, 6, 8],
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
    "iri-s": {
        "components": {
            "fantasy": [2, 7, 12, 15],
            "empathic_concern": [1, 5, 9, 11],
            "perspective_taking": [4, 10, 14, 16],
            "personal_distress": [3, 6, 8, 13],
        },
    },
    "erq": {
        "components": {
            "reappraisal": [1, 3, 5, 7, 8, 10],
            "suppression": [2, 4, 6, 9],
        },
        "metric": "mean",
    },
}
