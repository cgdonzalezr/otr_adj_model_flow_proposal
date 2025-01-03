# Copyright © 2023 by Boston Consulting Group. All rights reserved

import numpy as np
from collections import OrderedDict

# seed for reproducible results in logistic regression optimization
SEED = 42

# relevant columns from LexisNexis data set
COLUMNS_LOADED_LN = [
    "name",
    "b2bcnt2y",
    "model2score",
    "sbfecardcount",
]

# relevant columns from 2022 LexisNexis data set
COLUMNS_LOADED_LN_NEW = [
    "accountnumber",
    "b2bcnt2y",
    "model2score",
    "sbfecardcount",
]

# Model score columns
COLUMNS_MODEL_SCORE = {
    "ln": "model2score_ln",
    "fico": "pg_fico_score",
}

# Target column
COLUMN_TARGET = "is_bad"


# Columns for save for preprocessed sample
OUTPUT_COLUMNS_PREPROCESSED_SAMPLE = [
    'application_number',
    'ar_id',
    'booked',
    'decision_date',
    'number_of_trucks_c',
    'program_group',
    'product_type_c',
    'fraud_flag_c',
    'existing_exposure_check_c',
    'credit_line_requested_c',
    'years_in_business_c',
    'risk_grade_c',
    'security_deposit_c',
    'account_type_c',
    'credit_line_approved_c',
    'offer_name_wf_txt_c',
    'past_due_total',
    'current_ar_total',
    'pmts_amount',
    'days_to_90dpd',
    'chargeoff',
    'chargeoff_y1',
    'funding_type',
    'fico_score',
    'ln_score',
    'ln_segment',
    'is_bad',
]

# Columns to save for risk grade sample
OUTPUT_COLUMNS_RISK_SAMPLE = [
    "application_number",
    "ar_id",
    "program_group",
    "segment",
    "decision_date",
    "fico_score",
    "fico_score_z",
    "fico_score_filled",
    "fico_score_filled_z",
    "no_fico",
    "ln_score",
    "ln_score_z",
    "blended_score",
    "fico_score_guardrail_applied",
    "ln_score_guardrail_applied",
    "ln_impact",
    "fico_impact",
    "score_responsible_for_decline",
    "pred",
    "risk_grade_before_guardrail",
    "risk_grade",
    "risk_grade_granular",
    "confidence_level",
    "booked",
    "number_of_trucks_c",
    "is_bad",
]


# Year used for model development
MODEL_DEVELOPMENT_YEAR = 2021

# Quarter used for model application / testing
MODEL_APPLICATION_YEAR = 2022
MODEL_APPLICATION_QUARTER = 4

# Model deployment date
MODEL_DEPLOYMENT_DATE = "2024-02-16"
# MODEL_DEPLOYMENT_DATE = "2023-01-01"


# Model score valid ranges, to remove error codes
RANGE_LN_SCORE = [500, 900]
RANGE_FICO_SCORE = [300, 850]


# Segments for risk calibration
CONDITION_SBFE = {
    "column": "ln_segment",
    "allowed_values": ["sbfe"],
}
CONDITION_SBA = {
    "column": "ln_segment",
    "allowed_values": ["sba"],
}
CONDITION_LN_NOHIT = {
    "column": "ln_segment",
    "allowed_values": ["double_no_hit", "UNKNOWN"],
}

RISK_GRADE_SEGMENTS = {
    "sbfe_ln_and_fico": {
        "condition": CONDITION_SBFE,
        "model_cols": ["intercept", "ln_score", "fico_score_filled", "no_fico"],
        "train_filter": "booked",
        "core_model_vars": ["fico_score", "ln_score"],
    },  # LN SBFE hit, use blended score of LN and FICO
    "sba_ln_and_fico": {
        "condition": CONDITION_SBA,
        "model_cols": ["intercept", "ln_score", "fico_score_filled", "no_fico"],
        "train_filter": "booked",
        "core_model_vars": ["fico_score", "ln_score"],
    },  # LN SBFE hit, use blended score of LN and FICO
    "no_hit": {
        "condition": CONDITION_LN_NOHIT,
        "model_cols": ["intercept", "fico_score"],
        "train_filter": "booked & has_fico",
        "core_model_vars": ["fico_score"],
    },  # LN no-hit, use only FICO score
}

# Segments that we use to make up full dataset, one score per applications
SEGMENTS_PARTITION = ["sbfe_ln_and_fico", "sba_ln_and_fico", "no_hit"]

# Whether to normalize input scores before fitting logistic regression to generate blended score
NORMALIZE_SCORES = False

# score guard rails for risk grades
RG_GUARD_RAILS = {
    "fico_score": OrderedDict(
        {
            660: "2",
            630: "3",
            620: "4",
            600: "5",
            510: "6",
        }
    ),
    "ln_score": OrderedDict(
        {
            650: "2",
            630: "3",
            610: "4",
            570: "5",
            520: "6",
        }
    ),
}

# Risk grade labels and thresholds
# --------------------------------
RG_LABELS = [str(rg) for rg in range(1, 8)]
RG_THRESHOLDS = [
    0,
    *[np.exp((3 * i - 1) * np.log(0.02) / 19).round(3) for i in np.arange(6, 0, -1)],
    1,
]
RISK_GRADES = {
    label: {"lower": lower, "midpoint": (lower + upper) / 2, "upper": upper}
    for label, lower, upper in zip(RG_LABELS, RG_THRESHOLDS[:-1], RG_THRESHOLDS[1:])
}
RISK_GRADE_SCORING_MAP = {
    "segment_auc": {
        (0.0, 0.65): 0,
        (0.65, 0.7): 1,
        (0.7, 0.75): 2,
        (0.75, 1.0): 3,
    },
    "segment_historic_coverage": {
        (0.0, 0.25): 0,
        (0.25, 0.5): 1,
        (0.5, 0.75): 2,
        (0.75, 1.0): 3,
    },
    "rg_calibration": {
        (1, np.inf): -1,
        (0, 1): 0,
        (-np.inf, 0): 3,
    },
    "rg_approval_rate": {
        (0.0, 0.4): 0,
        (0.4, 0.5): 1,
        (0.5, 0.6): 2,
        (0.6, 0.7): 3,
        (0.7, 0.8): 4,
        (0.8, 0.9): 5,
        (0.9, 1.0): 6,
    },
    "rg_volume": {
        (0, 100): 0,
        (100, 200): 1,
        (200, 400): 2,
        (400, np.inf): 3,
    },
}
RISK_GRADE_CONFIDENCE_MAP = {
    "points": [-1, 11, 14, 18],
    "levels": ["Low", "Medium", "High"],
}

# Granular risk grade labels and thresholds
# -----------------------------------------
RG_LABELS_GRANULAR = [str(rg) + sub for rg in range(1, 8) for sub in ["a", "b", "c"]][:-1]
RG_THRESHOLDS_GRANULAR = [0] + [np.exp(i * np.log(0.02) / 19).round(3) for i in range(19, -1, -1)]

# Program names
# -------------
NASTC_PROGRAM_NAMES = ["NASTC - OTR", "NASTC - OTR - CA", "NASTC - OTR - CANADA"]
RTS_PROGRAM_NAMES = ["RTS Fleet One", "RTS Fleet One Crossroads Freight CT7"]
XPO_PROGRAM_NAMES = [
    "RXO FUEL CARD",
    "RXO Online",
    "XPO Logistics",
    "XPO Logistics EN",
    "XPO Logistics EN CANADA",
]
EDGE_PROGRAM_NAMES = [
    "Fleet One Edge LLC BOCA SG",
    "EDGE Plus Bundle",
    "FleetOne Edge Guaranteed Line",
    "Z Fleet One EDGE PLUS",
]
