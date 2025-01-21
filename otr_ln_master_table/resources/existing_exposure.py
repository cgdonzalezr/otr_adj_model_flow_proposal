import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

import model_training as mt
import config

np.random.seed(config.SEED)

def slice_to_dev_sample_ee(data: pd.DataFrame) -> pd.DataFrame:
    """Remove any rows to be excluded from development sample."""
    data = data.copy()  # Avoid changes outside function scope

    # Track filters impacting sample
    filters_applied = {"original_rows": len(data)}

    # Keep unique applications
    data = data.drop_duplicates(subset="application_number")
    filters_applied["unique_apps"] = len(data)

    # Keep just apps with existing exposure
    data = data.loc[data["existing_exposure"] == 1]
    filters_applied["remove_new_applications"] = len(data)

    # Remove apps with prepaid or deposit accounts
    data = data.loc[~data["account_type_c"].fillna("?").isin(["Prepay", "Deposit Account", "?"])]
    filters_applied["remove_prepaid_deposit"] = len(data)

    # Remove apps with credit line approved < 1
    data = data.loc[~(data["credit_line_approved_c"] < 1)]
    filters_applied["remove_credit_line_approved_lt_1"] = len(data)

    # Remove apps with security deposit
    data = data.loc[~data["security_deposit_c"]]
    filters_applied["remove_security_deposit"] = len(data)

    # Removed apps with credit line > 150,000$
    data = data.loc[data["credit_line_requested_c"] <= 1.5e5]
    filters_applied["remove_credit_line_gt_150k"] = len(data)

    # Remove apps flagged as fraud
    data = data.loc[~data["fraud_flag_c"]]
    filters_applied["remove_fraud_flag"] = len(data)

    # Remove first payment defaults
    data = data.loc[~mt.is_first_payment_default(data)]
    filters_applied["remove_first_payment_default"] = len(data)

    # Remove apps that are not child-funded
    data = data.loc[data["funding_type"] != "non_child_funded"]
    filters_applied["remove_non_child_funded"] = len(data)

    return data, filters_applied
