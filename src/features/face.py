# TODO: more explicit would be to use with column names instead of exclude additional columns
# TODO: maybe upsample for equidistant time points?

"""
cited after: Kappesser J. 2019 "The facial
expression of pain in humans considered from
a social perspective"
typically involve the lowering of the eyebrows, squeezing of the eyes, wrinkling of the nose, raising of the upper lip and
opening of the mouth (Craig KC, Prkachin KM, Grunau RE. 2011), which can be observed for
clinical and experimental pain (Williams ACdeC. 2002 Facial expression of pain: an
evolutionary account. Behav. Brain Sci. 25, 4.
(doi:10.1017/S0140525X02000080))
"""  # TODO REMOVE

import polars as pl

from src.features.scaling import scale_percent_to_decimal

SAMPLE_RATE = 10  # rounded up from 9.76 (0.76% coefficient of variation between trials)


def preprocess_face(df: pl.DataFrame) -> pl.DataFrame:
    df = scale_face(df)
    return df


def feature_face(df: pl.DataFrame) -> pl.DataFrame:
    df = df.drop(non_feature_columns)
    return df


def scale_face(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the facial expression columns to the range [0, 1] by dividing by 100.
    """
    return scale_percent_to_decimal(
        df,
        exclude_additional_columns=[
            "blink",
            "blinkrate",
            "interocular_distance",
        ],
    )


non_feature_columns = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "engagement",
    "valence",
    "sentimentality",
    "confusion",
    "neutral",
    "attention",
    "brow_raise",
    "chin_raise",
    "dimpler",
    "eye_closure",
    "eye_widen",
    "inner_brow_raise",
    "jaw_drop",
    "lip_corner_depressor",
    "lip_press",
    "lip_pucker",
    "lip_stretch",
    "lip_suck",
    "lid_tighten",
    "smile",
    "smirk",
    "blink",
    "blinkrate",
    "pitch",
    "yaw",
    "roll",
    "interocular_distance",
]
