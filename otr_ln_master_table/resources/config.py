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



CONDITIONS_NEW_MODEL = [
    (lambda df: (df['pg_required_c'] == True) & (df['sbfeaccountcount_ln'] >= 1) & (df['sbfehitindex_ln'] >= 2) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == True) & (df['sbfeaccountcount_ln'] >= 1) & (df['sbfehitindex_ln'] >= 2) & (df['fico_score'].isnull())),
    (lambda df: (df['pg_required_c'] == True) & (df['b2bcnt2y_ln'] >= 1) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == True) & (df['b2bcnt2y_ln'] >= 1) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].isnull())),
    (lambda df: (df['pg_required_c'] == True) & ((df['sbfeaccountcount_ln'] < 1) | (df['sbfeaccountcount_ln'].isnull())) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & ((df['sbfehitindex_ln'] <= 1) | (df['sbfehitindex_ln'].isnull())) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == True) & ((df['sbfeaccountcount_ln'] < 1) | (df['sbfeaccountcount_ln'].isnull())) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & ((df['sbfehitindex_ln'] <= 1) | (df['sbfehitindex_ln'].isnull())) & (df['fico_score'].isnull())),
    (lambda df: (df['pg_required_c'] == False) & (df['sbfeaccountcount_ln'] >= 1) & (df['sbfehitindex_ln'] >= 2)),
    (lambda df: (df['pg_required_c'] == False) & (df['b2bcnt2y_ln'] >= 1) & (df['sbfehitindex_ln'] == 1)),
    (lambda df: (df['pg_required_c'] == False) & ((df['sbfeaccountcount_ln'] < 1) | (df['sbfeaccountcount_ln'].isnull())) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & ((df['sbfehitindex_ln'] < 1) | (df['sbfehitindex_ln'].isnull()))),
#     (lambda df: (df['pg_required_c'] == True) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].notnull())), # merged with segment 5
#     (lambda df: (df['pg_required_c'] == True) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].isnull())), # merged with segment 6, no hits
    (lambda df: (df['pg_required_c'] == False) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].notnull())),
    (lambda df: (df['pg_required_c'] == False) & ((df['b2bcnt2y_ln'] < 1) | (df['b2bcnt2y_ln'].isnull())) & (df['sbfehitindex_ln'] == 1) & (df['fico_score'].isnull()))
]

CHOICES_NEW_MODEL = [
    'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
    'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
    'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
    'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
    'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit', 
    'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_no_hit', 
    'no_pg_and_1_plus_sbfe_tradeline',
    'no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline',
    'no_pg_no_sbfe_no_sba_no_fico',
#     'pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit', ## contradiction between hitindex and b2bcnt
#     'pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit', ## contradiction between hitindex and b2bcnt
    'no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit', ## contradiction between hitindex and b2bcnt, how can have fico
    'no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit' ## contradiction between hitindex and b2bcnt
]


CONDITION_PG_SBFE_FICO = {
    "column": "risk_grade_path",
    "allowed_values": ["pg_and_1_plus_sbfe_tradeline_and_fico_hit"],
}
CONDITION_PG_SBFE_NO_FICO = {
    "column": "risk_grade_path",
    "allowed_values": ["pg_and_1_plus_sbfe_tradeline_and_fico_no_hit"],
}
CONDITION_PG_SBA_FICO = {
    "column": "risk_grade_path",
    "allowed_values": ["pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit"],
}
CONDITION_PG_SBA_NO_FICO = {
    "column": "risk_grade_path",
    "allowed_values": ["pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit"],
}
CONDITION_PG_FICO_ONLY = {
    "column": "risk_grade_path",
    "allowed_values": ["pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit"],
}
CONDITION_PG_NO_HITS = {
    "column": "risk_grade_path",
    "allowed_values": ["pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_no_hit"],
}
CONDITION_NO_PG_SBFE = {
    "column": "risk_grade_path",
    "allowed_values": ["no_pg_and_1_plus_sbfe_tradeline"],
}
CONDITION_NO_PG_SBA = {
    "column": "risk_grade_path",
    "allowed_values": ["no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline"],
}
CONDITION_NO_PG_NO_HITS = {
    "column": "risk_grade_path",
    "allowed_values": ["no_pg_no_sbfe_no_sba_no_fico"],
}
# CONDITION_PG_FALSE_SBA_FICO_HIT = {
#     "column": "risk_grade_path",
#     "allowed_values": ["pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit"],
# }
# CONDITION_PG_FALSE_SBA_NO_HITS = {
#     "column": "risk_grade_path",
#     "allowed_values": ["pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit"],
# }
CONDITION_NO_PG_FALSE_SBA_FICO_HIT = {
    "column": "risk_grade_path",
    "allowed_values": ["no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_hit"],
}
CONDITION_NO_PG_FALSE_SBA_NO_HITS = {
    "column": "risk_grade_path",
    "allowed_values": ["no_pg_and_no_sbfe_tradeline_and_false_1_plus_sba_tradeline_and_fico_no_hit"],
}



RISK_GRADE_SEGMENTS_NEW_MODEL = {
    "pg_sbfe_ln_and_fico": {
        "condition": CONDITION_PG_SBFE_FICO,
        "model_cols": ["intercept", "ln_score", "fico_score_filled"],
        "train_filter": "booked",
        "core_model_vars": ["fico_score", "ln_score"],
    },  # LN SBFE hit, use blended score of LN and FICO
    "pg_sbfe_ln_only": {
        "condition": CONDITION_PG_SBFE_NO_FICO,
        "model_cols": ["intercept", "ln_score"],
        "train_filter": "booked",
        "core_model_vars": ["ln_score"],
    },  # LN SBFE hit, no FICO, use LN score only
    "pg_sba_ln_and_fico": {
        "condition": CONDITION_PG_SBA_FICO,
        "model_cols": ["intercept", "ln_score", "fico_score_filled"],
        "train_filter": "booked",
        "core_model_vars": ["fico_score", "ln_score"],
    },  # LN SBA hit, use blended score of LN and FICO
    "pg_sba_ln_only": {
        "condition": CONDITION_PG_SBA_NO_FICO,
        "model_cols": ["intercept", "ln_score"],
        "train_filter": "booked",
        "core_model_vars": ["ln_score"],
    },  # LN SBA hit, no FICO, use LN score only
    "pg_fico_only": {
        "condition": CONDITION_PG_FICO_ONLY,
        "model_cols": ["intercept", "fico_score"],
        "train_filter": "booked",
        "core_model_vars": ["fico_score"],
    },  # PG required, only FICO available
    "pg_no_hits": {
        "condition": CONDITION_PG_NO_HITS,
        "model_cols": ["intercept"],  # Intercept only model
        "train_filter": "booked",
        "core_model_vars": [], 
    }, # PG required, no hits, no FICO
    "no_pg_sbfe_ln_only": {
        "condition": CONDITION_NO_PG_SBFE,
        "model_cols": ["intercept", "ln_score"],
        "train_filter": "booked",
        "core_model_vars": ["ln_score"],
    },  # No PG, SBFE hit, use LN score only
    "no_pg_sba_ln_only": {
        "condition": CONDITION_NO_PG_SBA,
        "model_cols": ["intercept", "ln_score"],
        "train_filter": "booked",
        "core_model_vars": ["ln_score"],
    },  # No PG, SBA hit, use LN score only
    "no_pg_no_hits": {
        "condition": CONDITION_NO_PG_NO_HITS,
        "model_cols": ["intercept"], # Intercept only model
        "train_filter": "booked",
        "core_model_vars": [],
    },  # No PG,FICO hit
#     "pg_false_sba_fico_hit": {
#         "condition": CONDITION_PG_FALSE_SBA_FICO_HIT,
#         "model_cols": ["intercept"], # Intercept only model
#         "train_filter": "booked",
#         "core_model_vars": [],
#     },  # PG Hit 1, FICO hit
#     "pg_false_sba_no_hits": {
#         "condition": CONDITION_PG_FALSE_SBA_NO_HITS,
#         "model_cols": ["intercept"], # Intercept only model
#         "train_filter": "booked",
#         "core_model_vars": [],
#     },  # PG Hit 1, no hits
    "no_pg_false_sba_fico_hit": {
        "condition": CONDITION_NO_PG_FALSE_SBA_FICO_HIT,
        "model_cols": ["intercept"], # Intercept only model
        "train_filter": "booked",
        "core_model_vars": [],
    },  # PG Hit 1, FICO hit
    "no_pg_false_sba_no_hits": {
        "condition": CONDITION_NO_PG_FALSE_SBA_NO_HITS,
        "model_cols": ["intercept"], # Intercept only model
        "train_filter": "booked",
        "core_model_vars": [],
    },  # PG Hit 1, no hits
}

NORMALIZE_SCORES_NEW_MODEL = True


# Define the risk grade thresholds
NEW_PD_RISK_GRADE_THRESHOLDS = {
    "1a": (0.00, 0.0009),
    "1b": (0.0009, 0.0022),
    "1c": (0.0022, 0.0066),
    "2a": (0.0066, 0.0110),
    "2b": (0.0110, 0.0165),
    "2c": (0.0165, 0.0248),
    "3a": (0.0248, 0.0371),
    "3b": (0.0371, 0.0464),
    "3c": (0.0464, 0.0557),
    "4a": (0.0557, 0.0835),
    "4b": (0.0835, 0.1040),
    "4c": (0.1040, 0.1280),
    "5a": (0.1280, 0.1570),
    "5b": (0.1570, 0.1930),
    "5c": (0.1930, 0.2370),
    "6a": (0.2370, 0.2910),
    "6b": (0.2910, 0.3570),
    "6c": (0.3570, 0.5000),
    "7a": (0.5000, 0.8140),
    "7b": (0.8140, 1.0000),
}

# Define the approval/decline thresholds based on risk grade
APPROVAL_THRESHOLDS = {
    "1a": True,
    "1b": True,
    "1c": True,
    "2a": True,
    "2b": True,
    "2c": True,
    "3a": True,
    "3b": True,
    "3c": True,
    "4a": False,  # Decline starts here
    "4b": False,
    "4c": False,
    "5a": False,
    "5b": False,
    "5c": False,
    "6a": False,
    "6b": False,
    "6c": False,
    "7a": False,
    "7b": False,
}
