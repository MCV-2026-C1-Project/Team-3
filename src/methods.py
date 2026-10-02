from .color_histogram_retrieval import HistogramConfig


# Original individual-channel experiment retained as the personal RGB baseline.
RGB_G = HistogramConfig(
    name="Individual RGB-G (original experiment)",
    color_space="rgb",
    bins=16,
    sigma=0.5,
    metric="hellinger",
    channel_weights=(0.0, 1.0, 0.0),
)

# Configurations selected in the team's common notebook.
HSV_SELECTED = HistogramConfig(
    name="Team HSV robust selection",
    color_space="hsv",
    bins=32,
    sigma=1.0,
    metric="l1",
    channel_weights=(1 / 3, 1 / 3, 1 / 3),
)

LAB_SELECTED = HistogramConfig(
    name="Team Lab robust selection",
    color_space="lab",
    bins=64,
    sigma=0.0,
    metric="l1",
    channel_weights=(1 / 3, 1 / 3, 1 / 3),
)

HSV_BEST_OBSERVED = HistogramConfig(
    name="Team HSV best observed",
    color_space="hsv",
    bins=16,
    sigma=0.5,
    metric="l1",
    channel_weights=(1 / 3, 1 / 3, 1 / 3),
)

# Expanded-bin experiments
LAB_WEIGHTED_EQUAL = HistogramConfig(
    name="Improved Lab - simple",
    color_space="lab",
    bins=40,
    sigma=0.0,
    metric="l1",
    channel_weights=(1 / 3, 1 / 3, 1 / 3),  # transparent, equal-channel baseline
)

LAB_WEIGHTED = HistogramConfig(
    name="Improved Lab - tuned",
    color_space="lab",
    bins=48,
    sigma=0.5,
    metric="hellinger",
    # Downweight lightness and emphasize the two chromatic Lab axes.
    channel_weights=(0.125, 0.5, 0.375),
)

AUDIT_METHODS = (
    RGB_G,
    HSV_SELECTED,
    LAB_SELECTED,
    HSV_BEST_OBSERVED,
    LAB_WEIGHTED_EQUAL,
    LAB_WEIGHTED,
)

SUBMISSION_METHODS = {
    "method1": LAB_WEIGHTED_EQUAL,
    "method2": LAB_WEIGHTED,
}
