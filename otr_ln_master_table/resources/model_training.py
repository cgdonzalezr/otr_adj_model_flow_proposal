import pandas as pd
import numpy as np
from functools import partial
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.discrete.discrete_model import Logit, BinaryResultsWrapper
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from reject_inference import hard_cutoff_augmentation, fuzzy_augmentation, em_algorithm

import config

np.random.seed(config.SEED)

def is_first_payment_default(data: pd.DataFrame) -> pd.Series:
    """Determine whether a default occurred on the first payment."""
    return (
        (data.days_to_90dpd < 365)
        & (data.pmts_amount.fillna(0) < 100)
        & (data.past_due_total + data.current_ar_total > 100)
    )

def slice_to_dev_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Remove any rows to be excluded from development sample."""
    data = data.copy()  # Avoid changes outside function scope

    # Track filters impacting sample
    filters_applied = {"original_rows": len(data)}

    # Keep unique applications
    data = data.drop_duplicates(subset="application_number")
    filters_applied["unique_apps"] = len(data)

    # Remove apps failing existing exposure check
    data = data.loc[data["existing_exposure_check_c"] == "Passed"]
    filters_applied["remove_existing_exposure"] = len(data)

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
    data = data.loc[~is_first_payment_default(data)]
    filters_applied["remove_first_payment_default"] = len(data)

    # Remove apps that are not child-funded
    data = data.loc[data["funding_type"] != "non_child_funded"]
    filters_applied["remove_non_child_funded"] = len(data)

    return data, filters_applied

def filters_applied_chart(filters_applied_dev: Dict, filters_applied_test: Dict, filters_applied_scoring: Dict) -> None:
    """Measures the impact of filters on the data sample."""
    # Create a DataFrame with the filters applied
    filters_applied_df = pd.DataFrame(
        {
            "Development": filters_applied_dev,
            "Application": filters_applied_test,
            "Scoring": filters_applied_scoring,
        }
    ).T.fillna(0)

    # Transform the values to percentages
    filters_applied_df = filters_applied_df.div(filters_applied_df.sum(axis=1), axis=0) * 100

    # Plot the filters applied
    fig, ax = plt.subplots(figsize=(12, 6))
    filters_applied_df.plot(kind="bar", ax=ax)
    ax.set_title("Impact of Filters on Data Sample")
    ax.set_ylabel("Percentage of Applications")
    ax.set_xlabel("Sample")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.1f}%",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
            rotation = 90
        )

    ax.tick_params(axis='x', labelbottom=False, bottom=False)
    ax.set_xticklabels(filters_applied_df.index, y=-0.1) 
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xticklabels(filters_applied_df.index, rotation=0)

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def booked_rejected_pie_chart(data: pd.DataFrame) -> None:
    """
    Create a pie chart for booked and rejected transactions
    :param data: DataFrame with the data
    """
    data = data.copy()
    data['booked'] = data['booked'].map({True: 'Booked', False: 'Rejected'})
    fig, ax = plt.subplots(figsize=(6, 6))
    data.booked.value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], ax=ax)
    ax.set_title('Booked and Rejected Transactions')
    plt.show()


def default_pie_chart(data: pd.DataFrame) -> None:
    """
    Create a pie chart for defaulted and non-defaulted transactions
    :param data: DataFrame with the data
    """
    data = data.copy()
    data['booked'] = data['booked'].map({True: 'Booked', False: 'Rejected'})
    data['is_bad'] = data['is_bad'].map({True: 'Defaulted', False: 'Non-Defaulted'})
    data = data.loc[data['booked'] == 'Booked']
    fig, ax = plt.subplots(figsize=(6, 6))
    data.is_bad.value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], ax=ax)
    ax.set_title('Defaulted and Non-Defaulted Transactions')
    plt.show()



def build_model_development_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Filter for samples to be included in development data and restrict to dev time frame."""

    # restrict to development period
    data_dev_period = data.loc[pd.to_datetime(data["decision_date"]).dt.year == config.MODEL_DEVELOPMENT_YEAR]

    # Slice to development sample
    data_dev_sample = slice_to_dev_sample(data=data_dev_period)

    return data_dev_sample


def build_model_application_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Filter for samples to be included in test data and restrict to dev time frame."""

    # Slicing to application sample
    data_application_sample = data.loc[
        (pd.to_datetime(data["decision_date"]).dt.year == config.MODEL_APPLICATION_YEAR) & (
                    pd.to_datetime(data["decision_date"]).dt.quarter == config.MODEL_APPLICATION_QUARTER)
        ].pipe(slice_to_dev_sample)

    return data_application_sample


def build_model_scoring_sample(data: pd.DataFrame) -> pd.DataFrame:
    """Filter for samples to be included in scoring time frame."""

    # Slicing to scoring sample
    data_scoring_sample = data.loc[
        (pd.to_datetime(data["decision_date"]) > config.MODEL_DEPLOYMENT_DATE)
        ].pipe(slice_to_dev_sample)

    return data_scoring_sample


def add_filled_missing_scores_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Create new columns for filled scores and indicator variables."""

    data["no_fico"] = (~data["has_fico"]).astype(int)
    data["fico_score_filled"] = data["fico_score"].copy()
    data["fico_score_filled"] = data["fico_score_filled"].fillna(0)
    data["ln_score_filled"] = data["ln_score"].fillna(0)
    data["no_ln"] = data["ln_score"].isna().astype(int)

    return data

def risk_grade_estimate_preprocessing(data_in: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for risk grade estimation."""

    # copy data to avoid altering input
    data = data_in.copy()

    # Prepare scores for logistic regression
    data["has_fico"] = data["fico_score"].between(300, 850, inclusive="both")
    data["fico_score"] = data["fico_score"].where(data["has_fico"])
    data["intercept"] = 1  # Used as constant in logistic regression

    # fill missing scores and create indicator variables for NA
    data = add_filled_missing_scores_columns(data=data)

    return data


def filter_df_for_condition(df: pd.DataFrame, cond_dict: Dict) -> pd.DataFrame:
    """Filter data for a condition.

    The condition dictionary has the following form:
        cond_dict = {
            'column': 'col_to_filter_on',
            'allowed_values': ['allowed_val_1', 'allowed_val_2', ...]
        }
    """
    # make copy of input
    df_out = df.copy()

    # read filter config
    filter_col = cond_dict["column"]
    filter_vals = cond_dict["allowed_values"]

    # filter data
    mask = df_out[filter_col].isin(filter_vals)
    df_out = df_out[mask].copy()
    return df_out


def generate_model_parameters_df(models_dict: Dict[str, Dict]) -> pd.DataFrame:
    """Generate a dataframe that holds the parameters for the score blending.

    Parameters
    ----------
    models_dict : Dict[str, Dict]
        A dictionary where keys are segment names and values are dictionaries
        containing the model outputs from the `fit_risk_grades` function.

    Returns
    -------
    pd.DataFrame
        A dataframe containing the model parameters.
    """
    df_list = []
    # Process each segment's model output
    for segment, model_output in models_dict.items():
        # Try to extract a model from the known keys
        if model_output.get('method_name') == 'logistic_regression':
            model = model_output.get('financed_model', None)
        elif model_output.get('infered_model') is not None:
            model = model_output.get('infered_model', None)
        else:
            print(f"Skipping segment {segment} because it does not have parameters to extract")
            continue

        if model is None:
            print(f"Skipping segment {segment} because model is None")
            continue

        # Attempt to extract parameters in a robust way
        try:
            # If model.params is a pandas Series with a to_frame method, use it directly.
            if hasattr(model.params, "to_frame"):
                df = model.params.to_frame().transpose()
            else:
                # Otherwise, create a DataFrame from the parameters.
                df = pd.DataFrame([model.params])
        except Exception as e:
            print(f"Error processing model parameters for segment {segment}: {e}")
            continue

        df["segment"] = segment
        df_list.append(df)

    if not df_list:
        return pd.DataFrame()

    # Combine results into a single dataframe
    df_params = pd.concat(df_list, sort=False)
    # Ensure 'segment' is the first column in the dataframe
    cols = ["segment"] + [col for col in df_params.columns if col != "segment"]
    df_params = df_params[cols].reset_index(drop=True)

    return df_params
    


def fit_risk_grades(
    segment_name: str,
    data: pd.DataFrame,
    model_cols: List[str],
    core_model_vars: List[str],
    train_filter: pd.Series = None,
    NORMALIZE_SCORES: bool = True,
    reject_inference_method: Optional[str] = None,
    n_bins_parcelling: int = 10,
    prudence_factors_parcelling: Optional[List[int]] = None,

) -> Tuple[pd.DataFrame, pd.DataFrame, Logit, Dict]:
    """Assign risk grades to slice of data by fitting logistic regression.

    Parameters
    ----------
    segment_name
        Name of segment
    data
        Data to fit to risk grades
    model_cols
        List of columns used as independent variables in logistic regression
    core_model_vars
        List of original (score) variables used for modelling (w/o replacement of missing values)
    train_filter
        Filter applied to isolate training data from full segment
    reject_inference_method
        Method for reject inference: 'augmentation', 'fuzzy_augmentation', 'reclassification', 'twins', 'parcelling', 'em_algorithm', or None for basic logistic regression.
    n_bins_parcelling
         Number of bins for parcelling.
    prudence_factors_parcelling
         Prudence factors for parcelling.
   
    Returns
    -------
    data
        Copy of original data, with segment name, model prediction and risk grade
    metrics_confidence
        Metrics used for confidence assessment
    logistic_regressor
        Logistic regression model, fit to data
    variable_normalizations
        Normalizations applied to variables
    """

    # set train filter to in-sample filter if it is not provided
    if train_filter is None:
        train_filter = data["in_sample"]

    # Avoid changes outside function scope
    data = data.copy()

    # Normalizing independent variables via z-transform
    model_cols_norm = []
    if NORMALIZE_SCORES:
        variable_normalizations = {}
        for col in model_cols:
            std = data[col].std()
            if (std > 0) and (set(data[col].unique()) != {0, 1}):  # to skip the intercept and flags
                data[col + "_z"] = (data[col] - data[col].mean()) / std
                model_cols_norm.append(col + "_z")
                variable_normalizations[col] = {
                    "std": std,
                    "mean": data[col].mean(),
                }
            else:
                model_cols_norm.append(col)
    else:
        model_cols_norm = model_cols
        variable_normalizations = None

    assert data.loc[train_filter, "booked"].all()  # Do not train on non-booked data

    # Prepare data for reject inference methods
    xf = data.loc[train_filter, model_cols_norm].values  # Features of financed applicants
    xf = pd.DataFrame(xf, columns=model_cols_norm)

    yf = data.loc[train_filter, "is_bad"].astype(
        "bool"
    ).values  # Target variable (repayment status) of financed applicants
    yf = pd.Series(yf)
    
    xnf = data.loc[~train_filter, model_cols_norm].values  # Features of not financed applicants
    xnf = pd.DataFrame(xnf, columns=model_cols_norm)

    # Fit logistic regression or reject inference model based on input
    if reject_inference_method is None:
        logistic_regressor = Logit(
            endog=data.loc[train_filter, "is_bad"].astype("bool"),
            exog=data.loc[train_filter, model_cols_norm],
        ).fit(maxiter=100, method="bfgs")
        model_output = {
            'method_name': 'logistic_regression',
            'financed_model': logistic_regressor,
            'acceptance_model': None,
            'infered_model': None
        }

    elif reject_inference_method == "hard_cutoff_augmentation":
        model_output = hard_cutoff_augmentation(xf, xnf, yf)
    elif reject_inference_method == "fuzzy_augmentation":
        model_output = fuzzy_augmentation(xf, xnf, yf)
    elif reject_inference_method == "em_algorithm":
        model_output = em_algorithm(xf, xnf, yf)
    else:
        raise ValueError(
            f"Invalid reject_inference_method: {reject_inference_method}"
        )
    
    return model_output, variable_normalizations





def fit_models_for_segments(data: pd.DataFrame, RISK_GRADE_SEGMENTS, NORMALIZE_SCORES, reject_inference_method, n_bins_parcelling, prudence_factors_parcelling) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
    """Fit a logit model that matches the scores for each segments to the empirical bad rate."""

    # initialize output objects
    data_out_list, metrics_dict, models_dict, normalization_dict = [], {}, {}, {}

    # iterate over all segments
    for segment_name, params in RISK_GRADE_SEGMENTS.items():
        # fit risk grades for one segment
        model_i, variable_normalizations_i = fit_risk_grades(
            segment_name=segment_name,
            data=filter_df_for_condition(df=data, cond_dict=params["condition"]),
            model_cols=params["model_cols"],
            core_model_vars=params["core_model_vars"],
            train_filter=data.eval(params["train_filter"]),
            NORMALIZE_SCORES = NORMALIZE_SCORES,
            reject_inference_method= reject_inference_method,
            n_bins_parcelling=n_bins_parcelling,
            prudence_factors_parcelling=prudence_factors_parcelling
        )

        # store results
        models_dict[segment_name] = model_i
        normalization_dict[segment_name] = variable_normalizations_i

    # combine results

    return  models_dict, normalization_dict




def apply_fitted_models_to_data(
    data: pd.DataFrame,
    models_dict: Dict,
    normalization_dict: Dict,
    RISK_GRADE_SEGMENTS: Dict,
    NORMALIZE_SCORES: bool = True,
) -> pd.DataFrame:
    """Apply fitted logistic regression models to data."""
    data = data.copy()
    data_out = pd.DataFrame()  # Ensure data_out is initialized
    
    for segment_name, params in RISK_GRADE_SEGMENTS.items():
        data_i = filter_df_for_condition(df=data, cond_dict=params["condition"])
        
        if NORMALIZE_SCORES:
            model_cols_norm = []
            for col in params["model_cols"]:
                if normalization_dict.get(segment_name) and col in normalization_dict[segment_name]:
                    data_i[col + "_z"] = (
                        data_i[col] - normalization_dict[segment_name][col]["mean"]
                    ) / normalization_dict[segment_name][col]["std"]
                    model_cols_norm.append(col + "_z")
                else:
                    model_cols_norm.append(col)
        else:
            model_cols_norm = params["model_cols"]
        
        if len(data_i) > 0:
            model = models_dict[segment_name].get('financed_model') or models_dict[segment_name].get('infered_model')
            if model is None:
                print(f"Skipping segment {segment_name} because model is None")
                continue
            
            try:
                exog_data = data_i[model_cols_norm]
                
                # Ensure constant term is added if required
                if hasattr(model, 'params') and "const" in model.params.index:
                    if "const" not in exog_data.columns:
                        exog_data = sm.add_constant(exog_data, has_constant='add')
                
                # Debugging: Print shapes
                print(f"Segment: {segment_name}")
                print(f"Expected features in model: {model.params.shape}")
                print(f"Actual features in data: {exog_data.shape}")
                print(f"Columns in exog_data: {exog_data.columns.tolist()}")
                
                # Align columns with the model's expected features
                expected_cols = model.params.index.tolist()
                missing_cols = [col for col in expected_cols if col not in exog_data.columns]
                if missing_cols:
                    print(f"⚠️ Missing columns in segment {segment_name}: {missing_cols}")
                exog_data = exog_data.reindex(columns=expected_cols, fill_value=0)
                
                # Apply model
                data_i[f"pd_{segment_name}"] = model.predict(exog=exog_data)
            except Exception as e:
                print(f"Error applying model for segment {segment_name}: {e}")
                continue
            
        data_out = pd.concat([data_out, data_i], axis=0)
    
    return data_out




# Calculate segment statistics with approval/decline thresholds
def calculate_segment_statistics_with_approval(
    data: pd.DataFrame,
    RISK_GRADE_SEGMENTS: Dict,
    risk_grade_thresholds: Dict[str, Tuple[float, float]],
    approval_thresholds: Dict[str, bool],
    PG_SUBSEGMENTS: Dict = None,
) -> pd.DataFrame:
    """Calculate segment statistics with approval/decline decisions."""

    segment_stats = []

    # Iterate through Risk Grade Segments
    for segment_name, params in RISK_GRADE_SEGMENTS.items():
        # filter data for segment
        data_i = filter_df_for_condition(df=data, cond_dict=params["condition"])

        # If there are no PG Subsegments, compute the stats for the Risk Grade Segment
        if PG_SUBSEGMENTS is None:
            segment_stats.extend(_calculate_single_segment_stats(
                data=data_i,
                segment_name=segment_name,
                risk_grade_thresholds=risk_grade_thresholds,
                approval_thresholds=approval_thresholds,
                subsegment_name=None
            ))

        # If there are PG Subsegments, then we need to loop through them
        else:
            for subsegment_name, subsegment_params in PG_SUBSEGMENTS.items():
                data_j = filter_df_for_condition(df=data_i, cond_dict=subsegment_params["condition"])
                segment_stats.extend(_calculate_single_segment_stats(
                    data=data_j,
                    segment_name=segment_name, # keep original segment name
                    risk_grade_thresholds=risk_grade_thresholds,
                    approval_thresholds=approval_thresholds,
                    subsegment_name = subsegment_name
                ))

    segment_stats_df = pd.DataFrame(segment_stats)
    return segment_stats_df


def _calculate_single_segment_stats(
    data: pd.DataFrame,
    segment_name: str,
    risk_grade_thresholds: Dict[str, Tuple[float, float]],
    approval_thresholds: Dict[str, bool],
    subsegment_name: str = None,
) -> List[Dict]:

    total_rows = len(data)
    num_booked = data["booked"].sum()
    num_non_booked = total_rows - num_booked
    booked_is_bad = data[data["booked"] == 1]["is_bad"].sum()
    predicted_booked = 0
    predicted_booked_is_bad = 0
    predicted_declined = 0
    min_number_of_trucks = data["number_of_trucks_c"].min()
    max_number_of_trucks = data["number_of_trucks_c"].max()
    avg_number_of_trucks = data["number_of_trucks_c"].mean()
    min_years_in_business = data["years_in_business_num_c"].min()
    max_years_in_business = data["years_in_business_num_c"].max()
    avg_years_in_business = data["years_in_business_num_c"].mean()

    for risk_grade, (lower_bound, upper_bound) in risk_grade_thresholds.items():
        # Count predicted booked and declined based on risk grade and approval threshold
        mask = (data[f"pd_{segment_name}"] >= lower_bound) & (data[f"pd_{segment_name}"] < upper_bound)
        if approval_thresholds[risk_grade]:
            predicted_booked += mask.sum()
            predicted_booked_is_bad += data[mask]["is_bad"].sum()
        else:
            predicted_declined += mask.sum()

    output = {
            "segment_name": segment_name,
            "total_rows": total_rows,
            "num_booked": num_booked,
            "num_non_booked": num_non_booked,
            "booked_is_bad": booked_is_bad,
            "predicted_booked": predicted_booked,
            "predicted_booked_is_bad": predicted_booked_is_bad,
            "predicted_declined": predicted_declined,
            "min_number_of_trucks": min_number_of_trucks,
            "max_number_of_trucks": max_number_of_trucks,
            "avg_number_of_trucks": avg_number_of_trucks,
            "min_years_in_business": min_years_in_business,
            "max_years_in_business": max_years_in_business,
            "avg_years_in_business": avg_years_in_business,
        }

    if subsegment_name:
        output["subsegment_name"] = subsegment_name

    return [output]


# Create a function to apply the conditions and choices to a dataframe
def apply_risk_grade_path(
        df: pd.DataFrame,
        conditions: List,
        choices: List,
        ):
  """Applies risk grade path logic to a given dataframe."""
  evaluated_conditions = [condition(df) for condition in conditions]
  df['risk_grade_path'] = np.select(evaluated_conditions, choices, default="NAN")
  return df


def assign_risk_grade(probability, thresholds):
    """Assigns a risk grade based on probability and thresholds."""
    for grade, (lower, upper) in thresholds.items():
        if lower <= probability < upper:
            return grade
    return "Unknown"


def predict_with_model(df, parameters_df, segment):
    """Predicts probabilities based on model parameters."""
    params = parameters_df[parameters_df['segment'] == segment].iloc[0]
    intercept = params['intercept']
    ln_score_coeff = params['ln_score_z']
    fico_score_coeff = params['fico_score_filled_z']

    if pd.isna(ln_score_coeff) and pd.isna(fico_score_coeff):
        return np.exp(intercept) / (1 + np.exp(intercept))

    if pd.isna(fico_score_coeff):  # ln_only model
        linear_predictor = intercept + ln_score_coeff * df['ln_score_z']
    elif pd.isna(ln_score_coeff):  # fico_only model
        fico_score_col = 'fico_score_z' if segment == 'pg_fico_only' else 'fico_score_filled_z'
        linear_predictor = intercept + fico_score_coeff * df[fico_score_col]
    else:  # ln_and_fico model
        fico_score_col = 'fico_score_z' if segment == 'pg_fico_only' else 'fico_score_filled_z'
        linear_predictor = intercept + ln_score_coeff * df['ln_score_z'] + fico_score_coeff * df[fico_score_col]

    return np.exp(linear_predictor) / (1 + np.exp(linear_predictor))



def predict_and_assign_risk_grade_for_segment(segment, parameters, fico_scores, ln_scores, normalization_dict, new_pd_risk_grade_thresholds):
         """
         Generates predictions and assigns risk grades for the given segment across specified fico and ln_score ranges.
         """
         params = parameters[parameters['segment'] == segment].iloc[0].to_dict()
         predictions = None
         
         # Normalization inside the loop
         normalized_fico_scores = fico_scores
         normalized_fico_scores_filled = fico_scores
         normalized_ln_scores = ln_scores
         
         if normalization_dict.get(segment) is not None:
             if "fico_score" in normalization_dict[segment]:
                 mean = normalization_dict[segment]["fico_score"]["mean"]
                 std = normalization_dict[segment]["fico_score"]["std"]
                 if std != 0:
                     normalized_fico_scores = [(score - mean) / std for score in fico_scores]
         
             if "fico_score_filled" in normalization_dict[segment]: # Normalize filled scores separately
                  mean = normalization_dict[segment]["fico_score_filled"]["mean"]
                  std = normalization_dict[segment]["fico_score_filled"]["std"]
                  if std != 0:
                       normalized_fico_scores_filled = [(score - mean) / std for score in fico_scores]
         
             if "ln_score" in normalization_dict[segment]:
                  mean = normalization_dict[segment]["ln_score"]["mean"]
                  std = normalization_dict[segment]["ln_score"]["std"]
                  if std != 0:
                      normalized_ln_scores = [(score - mean) / std for score in ln_scores]
         
         if pd.isna(params['ln_score_z']) and pd.isna(params['fico_score_filled_z']):  # Constant only model
             predictions = predict_with_model(pd.DataFrame(), parameters, segment)
             print(f"Predictions for segment: {segment}")
             print(f"Constant prediction: {predictions}")
         
         
         elif pd.isna(params['fico_score_filled_z']):  # ln_score only
             predictions = predict_with_model(pd.DataFrame({'ln_score_z': normalized_ln_scores}), parameters, segment)
             print(f"Predictions for segment: {segment}")
             for i, prediction in enumerate(predictions):
                 print(f"{normalized_ln_scores[i]}: {prediction}")  # Use normalized ln_scores
         
         
         
         elif pd.isna(params['ln_score_z']):  # fico_score only
             fico_score_col = 'fico_score_z' if segment == 'pg_fico_only' else 'fico_score_filled_z' # Make sure column names align in parameters and normalization_dict
             predictions = predict_with_model(pd.DataFrame({fico_score_col: normalized_fico_scores}), parameters, segment)
             print(f"Predictions for segment: {segment}")
             print(predictions)
         
         
         
         else:  # Both ln_score and fico_score
             predictions = pd.DataFrame(index=normalized_fico_scores_filled, columns=normalized_ln_scores)
             for fico in normalized_fico_scores_filled:
                 for ln in normalized_ln_scores:
                     fico_col = 'fico_score_z' if segment == 'pg_fico_only' else 'fico_score_filled_z'
                     data = {fico_col: [fico], 'ln_score_z': [ln]}
                     prediction = predict_with_model(pd.DataFrame(data), parameters, segment)
                     predictions.loc[fico, ln] = prediction.iloc[0]
                     
             predictions.index = list(zip(fico_scores, normalized_fico_scores_filled))
             predictions.columns = list(zip(ln_scores, normalized_ln_scores))
         
             print(f"Predictions for segment: {segment}")
             print(predictions)
         
         
         # Risk Grade Matrix/Series Creation
         if predictions is not None:
             if isinstance(predictions, pd.Series):
                 risk_grades = predictions.apply(lambda x: assign_risk_grade(x, new_pd_risk_grade_thresholds))
                 print(f"\nRisk Grade Series for segment: {segment}")
                 print(risk_grades)
             elif isinstance(predictions, pd.DataFrame):
                 risk_grade_matrix = predictions.applymap(lambda x: assign_risk_grade(x, new_pd_risk_grade_thresholds))
                 print(f"\nRisk Grade Matrix for segment: {segment}")
                 print(risk_grade_matrix)
         
             print("-" * 50)



def compute_decile_table(
        df: pd.DataFrame,
        y_pred_proba: str,
        target: str,
        num_bins: int,
        segment: str
) -> pd.DataFrame:
    """
    Computes a decile table for a given DataFrame containing predictions and target variable.

    This function takes a pandas DataFrame (`df`) containing columns for the model's predicted probability (`y_pred_proba`), 
    target variable (`target`), decision date (`decision_date`), and segment (`segment`). It also takes arguments for the 
    number of deciles (`num_bins`) and the target segment (`segment`) to filter by.

    The function performs the following steps:

    1. Filters data by the specified time range and segment.
    2. Sorts data by the predicted probability in descending order.
    3. Assigns deciles (groups of equal size) to each data point based on the predicted probability ranking.
    4. Calculates the total event rate for the entire dataset.
    5. Groups data by decile and calculates various metrics:
        - Count of accounts in each decile
        - Sum of target variable (defaults) in each decile
        - Average predicted probability in each decile
        - Minimum and maximum predicted probability range for each decile
    6. Calculates cumulative counts, percentages, default rates, gains, and lifts for each decile.
    7. Selects and returns a DataFrame containing the desired columns for display.

    Args:
        df (pandas.DataFrame): The input DataFrame containing model predictions and target variable.
        y_pred_proba (str): The column name containing the model's predicted probability.
        target (str): The column name containing the target variable.
        num_bins (int): The number of deciles to create.
        segment (str): The segment to filter data by.

    Returns:
        pandas.DataFrame: A DataFrame containing the decile table with various metrics.
    """
    
    binded_df = df.copy() # Avoid high fragmentation

    # Filter data by segment
    if segment != None:
        binded_df = binded_df[binded_df['risk_grade_path'] == segment]

    # Calculate metrics
    binded_df = binded_df.sort_values(y_pred_proba, ascending = False)
    binded_df["DECILE"] = pd.qcut(binded_df[y_pred_proba].rank(method='first'), q = num_bins, labels = list(range(num_bins, 0, -1)))
    binded_df["TOTAL_EVENT_RATE"] =  binded_df[target].mean()

    # Calculate the number of accounts and default accounts for each decile
    decile_df = binded_df.groupby("DECILE", observed=False).agg(
        COUNT=(y_pred_proba, "count"),
        DEFAULT=(target, "sum"),
        TOTAL_EVENT_RATE=("TOTAL_EVENT_RATE", "mean"),
    ).reset_index()

    # Calculate the predicted probability range for each decile
    decile_df["PROB_RANGE"] = binded_df.groupby("DECILE", observed=False)[y_pred_proba].agg(["min", "max"]).apply(lambda x: f"({x['min']:.4f} - {x['max']:.4f}]", axis=1).values
    decile_df["AVG_PROB"] = binded_df.groupby("DECILE", observed=False)[y_pred_proba].agg(["mean"]).values
    decile_df["AVG_PROB"] = round(decile_df["AVG_PROB"], 4)

    # sort dataframes by decile
    decile_df = decile_df.sort_values(by="DECILE", ascending=False)

    # Calculate the cumulative number of accounts and default accounts
    decile_df["CUM_COUNT"] = round(decile_df["COUNT"].cumsum(),0)
    decile_df["CUM_DEFAULT"] = decile_df["DEFAULT"].cumsum()

    # Calculate the cumulative percentage of accounts and default accounts
    decile_df["CUM_PCT_COUNT"] = round(decile_df["CUM_COUNT"] / decile_df["COUNT"].sum() * 100, 3)
    decile_df["CUM_PCT_DEFAULT"] = round((decile_df["CUM_DEFAULT"] / decile_df["DEFAULT"].sum()), 3)

    # Calculate the default rate for each decile
    decile_df["DEFAULT_RATE"] = round((decile_df["DEFAULT"] / decile_df["COUNT"]), 3)

    # Calculate gain for each decile
    decile_df["GAIN"] = round((decile_df["CUM_DEFAULT"] / binded_df[target].sum()), 3)

    # Calculate the lift for each decile
    decile_df["LIFT"] = round((decile_df["GAIN"]*100) / (decile_df["CUM_PCT_COUNT"]), 3)

    cols_to_display = [
        "DECILE", "PROB_RANGE", "AVG_PROB", "COUNT", "DEFAULT_RATE", "TOTAL_EVENT_RATE", "CUM_COUNT", "CUM_DEFAULT", "CUM_PCT_DEFAULT",
        "GAIN", "LIFT"
    ]

    decile_df = decile_df[cols_to_display]
    # st.write(binded_df[binded_df['is_bad'] == True].head())
    # st.write(decile_df)
    return decile_df
        





def lift_chart_plot(
    plot_name: str,
    decile_df: pd.DataFrame,
    x_axis: str
):
    """
    Generates a lift chart visualizing the distribution of observations, 
    default rates, and average predicted probabilities across deciles.

    This function takes a plot name (`plot_name`) and a pandas DataFrame (`decile_df`) containing 
    decile information as input. The DataFrame is expected to have columns for decile index, 
    predicted probability range (x_axis), average predicted probability (`AVG_PROB`), 
    number of observations (`COUNT`), default rate (`DEFAULT_RATE`), and total event rate.

    The function creates a twin-axis plot to visualize the following:

    1. Bar chart representing the number of observations in each decile.
    2. Line plot showing the default rate for each decile.
    3. Line plot (dashed) representing the overall event rate for the entire dataset.
    4. Line plot showing the average predicted probability for each decile.

    Args:
        plot_name (str): The name to use for the chart title.
        decile_df (pandas.DataFrame): A DataFrame containing decile data.
        x_axis (str): The name of variable to use as x-axis

    Returns:
        None - The function creates the lift chart and displays it using Streamlit (`st.pyplot`).
    """
    event_rate = decile_df["TOTAL_EVENT_RATE"].astype(float).mean()

    # Build event rate plot
    fig = plt.figure(figsize=(12,4))

    # Plot barplot containing number of observations
    plt.bar(decile_df.index, decile_df["COUNT"], color="lightgray")

    # Add ticks & laels to axis
    plt.xlabel("Model Sorted Predictions (Low → High)")
    plt.ylabel("# Observations")
    plt.xticks(decile_df.index, decile_df[x_axis], rotation=45, ha='right', rotation_mode='anchor')
    plt.title(f"Actual vs. Predicted Lift Chart - {plot_name}")

    # Mirror plot and add event rates
    plt2 = plt.twinx()
    plt2.set_ylabel("Event rate")
    plt2.set_ylim(ymin=0, ymax=max(decile_df["AVG_PROB"].max(), decile_df["DEFAULT_RATE"].max()) + 0.05)
    plt2.set_yticks(np.arange(0, max(decile_df["AVG_PROB"].max(), decile_df["DEFAULT_RATE"].max()) + 0.05, step=0.05))
    plt2.plot(
        decile_df.index, decile_df["DEFAULT_RATE"], label="event_rate", marker="o"
    )

    # add average prediction
    plt2.plot(
        decile_df.index,
        decile_df["AVG_PROB"],
        label="average_prediction",
        marker="x",
        linestyle=":",
        color="black"
    )

    # Add global event rate as baseline
    plt2.plot(
        [min(decile_df.index) - 1, max(decile_df.index) + 1],
        [event_rate, event_rate],
        color="darkgrey",
        lw=1,
        linestyle="--",
        label=f"total_event_rate\n({'{:.1%}'.format(event_rate)})",
    )
    plt2.legend(loc=0)
    plt2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

    plt2.yaxis.grid(False)
    plt2.set_xlim([min(decile_df.index) - 0.5, max(decile_df.index) + 0.5])

    fig.show()




def compute_risk_grade_table(
        df: pd.DataFrame,
        y_pred_proba: str,
        target: str,
        num_bins: int,
        segment: str,
        risk_grade_column: str
) -> pd.DataFrame:
    """
    This function computes a risk grade table for a given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        y_pred_proba (str): The name of the column containing the predicted probability of the target event.
        target (str): The name of the column containing the target variable (e.g., default flag).
        num_bins (int): The number of bins to use for discretizing the predicted probability.
        segment (str): The segment filter to apply to the data (optional).
        risk_grade_column (str): The name of the column containing the risk grade information.

    Returns:
        pd.DataFrame: A DataFrame containing the risk grade table with the following columns:
            - risk_grade_column: The original risk grade values.
            - AVG_PROB: The average predicted probability for each risk grade.
            - COUNT: The number of accounts in each risk grade.
            - DEFAULT_RATE: The default rate (proportion of defaults) for each risk grade.
            - TOTAL_EVENT_RATE: The overall event rate (e.g., average target value) for the entire dataset.
            - CUM_COUNT: The cumulative number of accounts up to each risk grade.
            - CUM_DEFAULT: The cumulative number of defaults up to each risk grade.
            - CUM_PCT_COUNT: The cumulative percentage of accounts up to each risk grade.
            - CUM_PCT_DEFAULT: The cumulative percentage of defaults up to each risk grade.
            - GAIN: The proportion of total defaults captured up to each risk grade.
            - LIFT: The lift ratio for each risk grade (gain divided by cumulative percentage of accounts).
    """
    
    binded_df = df.copy() # Avoid high fragmentation

    # Filter data by segment
    binded_df = binded_df[binded_df['risk_grade_path'] == segment]

    # Calculate metrics
    binded_df = binded_df.sort_values(y_pred_proba, ascending = False)
    binded_df["TOTAL_EVENT_RATE"] =  binded_df[target].mean()

    # Calculate the number of accounts and default accounts for each decile
    risk_grade_df = binded_df.groupby(risk_grade_column, observed=False).agg(
        COUNT=(y_pred_proba, "count"),
        DEFAULT=(target, "sum"),
        TOTAL_EVENT_RATE=("TOTAL_EVENT_RATE", "mean"),
    ).reset_index()

    # Calculate the avg predicted probability
    risk_grade_df["AVG_PROB"] = binded_df.groupby(risk_grade_column, observed=False)[y_pred_proba].agg(["mean"]).values
    risk_grade_df["AVG_PROB"] = round(risk_grade_df["AVG_PROB"], 4)

    # sort dataframes by decile
    risk_grade_df = risk_grade_df.sort_values(by=risk_grade_column, ascending=False)

    # Calculate the cumulative number of accounts and default accounts
    risk_grade_df["CUM_COUNT"] = round(risk_grade_df["COUNT"].cumsum(),0)
    risk_grade_df["CUM_DEFAULT"] = risk_grade_df["DEFAULT"].cumsum()

    # Calculate the cumulative percentage of accounts and default accounts
    risk_grade_df["CUM_PCT_COUNT"] = round(risk_grade_df["CUM_COUNT"] / risk_grade_df["COUNT"].sum() * 100, 3)
    risk_grade_df["CUM_PCT_DEFAULT"] = round((risk_grade_df["CUM_DEFAULT"] / risk_grade_df["DEFAULT"].sum()), 3)

    # Calculate the default rate for each decile
    risk_grade_df["DEFAULT_RATE"] = round((risk_grade_df["DEFAULT"] / risk_grade_df["COUNT"]), 3)

    # Calculate gain for each decile
    risk_grade_df["GAIN"] = round((risk_grade_df["CUM_DEFAULT"] / binded_df[target].sum()), 3)

    # Calculate the lift for each decile
    risk_grade_df["LIFT"] = round((risk_grade_df["GAIN"]*100) / (risk_grade_df["CUM_PCT_COUNT"]), 3)

    cols_to_display = [
        risk_grade_column, "AVG_PROB", "COUNT", "DEFAULT_RATE", "TOTAL_EVENT_RATE", "CUM_COUNT", "CUM_DEFAULT", "CUM_PCT_DEFAULT",
        "GAIN", "LIFT"
    ]

    risk_grade_df = risk_grade_df[cols_to_display]

    return risk_grade_df




def generate_lift_chart(data, title, score_column, target_column, n_bins, segment, chart_type, risk_grade_thresholds = None, risk_grade_column = None):
    """
    Generates a lift chart based on specified parameters.
    """
    if chart_type == 'decile':
        table = compute_decile_table(data, score_column, target_column, n_bins, segment)
        x_axis_column = 'PROB_RANGE'
    elif chart_type == 'risk_grade':
          data['risk_grades'] = data[data['booked']][score_column].apply(lambda x: assign_risk_grade(x, risk_grade_thresholds))
          table = compute_risk_grade_table(data, score_column, target_column, n_bins, segment, risk_grade_column)
          x_axis_column = risk_grade_column
    else:
        print(f"Invalid chart type: {chart_type}. Skipping chart.")
        return
    
    lift_chart_plot(title, table, x_axis_column)


def generate_all_lift_charts(data_test_scored, data_scoring_scored, new_pd_risk_grade_thresholds):
    """
    Generates all lift charts using the reusable generate_lift_chart function.
    """
    chart_params = [
      # pd_pg_sbfe_ln_and_fico
        {
            'title': 'Test Data - pd_pg_sbfe_ln_and_fico - Decile',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
            'chart_type': 'decile',
        },
        {
            'title': 'Test Data - pd_pg_sbfe_ln_and_fico - Decile',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_sbfe_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
            'chart_type': 'decile',
        },
        {
            'title': 'Scoring Data - pd_pg_sbfe_ln_and_fico - Decile',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
            'chart_type': 'decile',
        },

          {
            'title': 'Test Data - pd_pg_sbfe_ln_and_fico - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
        },
        
       {
            'title': 'Test Data - pd_pg_sbfe_ln_and_fico - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_sbfe_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
        },
            {
            'title': 'Scoring Data - pd_pg_sbfe_ln_and_fico - Risk Grade',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_hit') & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
        },

         # pd_pg_sbfe_ln_only
        {
            'title': 'Test Data - pd_pg_sbfe_ln_only - Decile',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
             'chart_type': 'decile',
        },
        {
            'title': 'Test Data - pd_pg_sbfe_ln_only - Decile',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_sbfe_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
            'chart_type': 'decile',
        },
        {
            'title': 'Scoring Data - pd_pg_sbfe_ln_only - Decile',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
              'chart_type': 'decile',
        },
        {
            'title': 'Test Data - pd_pg_sbfe_ln_only - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
            'chart_type': 'risk_grade',
            'risk_grade_thresholds': new_pd_risk_grade_thresholds,
            'risk_grade_column': 'risk_grades'
            
        },
       {
            'title': 'Test Data - pd_pg_sbfe_ln_only - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_sbfe_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
        },
           {
            'title': 'Scoring Data - pd_pg_sbfe_ln_only - Risk Grade',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit') & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sbfe_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
        },

       # pd_pg_sba_ln_and_fico
       {
            'title': 'Test Data - pd_pg_sba_ln_and_fico - Decile',
             'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
             'score_column': 'pd_pg_sba_ln_and_fico',
             'target_column': 'is_bad',
             'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
             'chart_type': 'decile',
        },
        {
             'title': 'Test Data - pd_pg_sba_ln_and_fico - Decile',
              'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
              'score_column': 'pd_pg_sba_ln_and_fico',
              'target_column': 'is_bad',
              'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
            'chart_type': 'decile',
        },
        {
            'title': 'Scoring Data - pd_pg_sba_ln_and_fico - Decile',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sba_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
             'chart_type': 'decile',
        },
         {
            'title': 'Test Data - pd_pg_sba_ln_and_fico - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sba_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
            
         },
           {
            'title': 'Test Data - pd_pg_sba_ln_and_fico - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_sba_ln_and_fico',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
            
        },
           {
            'title': 'Scoring Data - pd_pg_sba_ln_and_fico - Risk Grade',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit') & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
             'score_column': 'pd_pg_sba_ln_and_fico',
             'target_column': 'is_bad',
             'n_bins': 20,
            'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit',
            'chart_type': 'risk_grade',
            'risk_grade_thresholds': new_pd_risk_grade_thresholds,
            'risk_grade_column': 'risk_grades'
           },
        
        # pd_pg_sba_ln_only
           {
             'title': 'Test Data - pd_pg_sba_ln_only - Decile',
             'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
              'score_column': 'pd_pg_sba_ln_only',
             'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
             'chart_type': 'decile',
        },
        # {
        #      'title': 'Test Data - pd_pg_sba_ln_only - Decile',
        #     'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
        #     'score_column': 'pd_pg_sba_ln_only',
        #     'target_column': 'is_bad',
        #     'n_bins': 20,
        #      'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
        #      'chart_type': 'decile',
        # },
         {
            'title': 'Scoring Data - pd_pg_sba_ln_only - Decile',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sba_ln_only',
            'target_column': 'is_bad',
             'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
            'chart_type': 'decile',
           },
          {
            'title': 'Test Data - pd_pg_sba_ln_only - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sba_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
            'chart_type': 'risk_grade',
            'risk_grade_thresholds': new_pd_risk_grade_thresholds,
            'risk_grade_column': 'risk_grades'
          },
        #   {
        #     'title': 'Test Data - pd_pg_sba_ln_only - Risk Grade',
        #    'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
        #    'score_column': 'pd_pg_sba_ln_only',
        #    'target_column': 'is_bad',
        #    'n_bins': 20,
        #     'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
        #     'chart_type': 'risk_grade',
        #     'risk_grade_thresholds': new_pd_risk_grade_thresholds,
        #    'risk_grade_column': 'risk_grades'
        # },
           {
            'title': 'Scoring Data - pd_pg_sba_ln_only - Risk Grade',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit') & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_sba_ln_only',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit',
            'chart_type': 'risk_grade',
           'risk_grade_thresholds': new_pd_risk_grade_thresholds,
           'risk_grade_column': 'risk_grades'
         },


      # pd_pg_fico_only
       {
            'title': 'Test Data - pd_pg_fico_only - Decile',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_fico_only',
            'target_column': 'is_bad',
            'n_bins': 20,
           'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
           'chart_type': 'decile',
       },
        {
            'title': 'Test Data - pd_pg_fico_only - Decile',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
           'score_column': 'pd_pg_fico_only',
           'target_column': 'is_bad',
           'n_bins': 20,
           'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
            'chart_type': 'decile',
       },
        {
            'title': 'Scoring Data - pd_pg_fico_only - Decile',
             'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
             'score_column': 'pd_pg_fico_only',
            'target_column': 'is_bad',
           'n_bins': 20,
            'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
             'chart_type': 'decile',
        },
        {
           'title': 'Scoring Data - pd_pg_fico_only - Decile',
           'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_fico_only',
            'target_column': 'is_bad',
           'n_bins': 20,
           'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
            'chart_type': 'decile',
       },
      {
            'title': 'Test Data - pd_pg_fico_only - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_fico_only',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
             'chart_type': 'risk_grade',
             'risk_grade_thresholds': new_pd_risk_grade_thresholds,
             'risk_grade_column': 'risk_grades'
        },
         {
            'title': 'Test Data - pd_pg_fico_only - Risk Grade',
            'data': data_test_scored[(data_test_scored['booked']) & (data_test_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit') & (data_test_scored['pg_subsegment'] == 'pg_should_not_be_required')],
            'score_column': 'pd_pg_fico_only',
            'target_column': 'is_bad',
            'n_bins': 20,
            'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
            'chart_type': 'risk_grade',
           'risk_grade_thresholds': new_pd_risk_grade_thresholds,
           'risk_grade_column': 'risk_grades'
       },
          {
            'title': 'Scoring Data - pd_pg_fico_only - Risk Grade',
            'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit') & (data_scoring_scored['pg_subsegment'] == 'pg_should_be_required')],
            'score_column': 'pd_pg_fico_only',
            'target_column': 'is_bad',
            'n_bins': 20,
             'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
            'chart_type': 'risk_grade',
           'risk_grade_thresholds': new_pd_risk_grade_thresholds,
            'risk_grade_column': 'risk_grades'
       },
    #     {
    #        'title': 'Scoring Data - pd_pg_fico_only - Risk Grade',
    #        'data': data_scoring_scored[(data_scoring_scored['booked']) & (data_scoring_scored['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit') & (data_scoring_scored['pg_subsegment'] == 'pg_should_not_be_required')],
    #        'score_column': 'pd_pg_fico_only',
    #        'target_column': 'is_bad',
    #        'n_bins': 20,
    #         'segment': 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit',
    #         'chart_type': 'risk_grade',
    #         'risk_grade_thresholds': new_pd_risk_grade_thresholds,
    #         'risk_grade_column': 'risk_grades'
    #    },
    ]

    for params in chart_params:
          generate_lift_chart(**params)




def analyze_booked_rates_for_segments(data_dev_scored):
    """Analyzes booked rates for different segments based on FICO and LN score bands."""
    
    print("Analyzing booked rates for segment: pg_and_1_plus_sbfe_tradeline_and_fico_hit")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_hit')]

    # Define fico score bands and ln_score ranges
    fico_bands = [580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840]
    fico_ranges = list(zip(fico_bands[:-1], fico_bands[1:]))
    fico_ranges.append((fico_bands[-1], 900))
    ln_bands = [500, 550, 600, 650, 700, 750, 800, 850, 900]
    ln_ranges = list(zip(ln_bands[:-1], ln_bands[1:]))
    ln_ranges.append((ln_bands[-1], 900))

    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=fico_ranges, columns=ln_ranges)  # Use ln_ranges directly for columns
    results_matrix = results_matrix.fillna('')
    total_test = 0
    # Iterate through fico bands and ln_score ranges
    for i in range(len(fico_ranges)):
        for j in range(len(ln_ranges)):  # Iterate through all ln_ranges
            fico_lower = fico_ranges[i][0]
            fico_upper = fico_ranges[i][1]
            ln_lower = ln_ranges[j][0]
            ln_upper = ln_ranges[j][1]

            filtered_df = data[
                (data['fico_score'] >= fico_lower) &
                (data['fico_score'] < fico_upper) &
                (data['ln_score'] >= ln_lower) &
                (data['ln_score'] < ln_upper)
            ]

            total_rows = len(filtered_df)
            if total_rows > 0:
                total_test += total_rows
                booked_rows = filtered_df['booked'].sum()
                booked_rate = (booked_rows / total_rows) * 100
                results_matrix.iloc[i, j] = f"{total_rows} ({booked_rate:.1f}%)"
            else:
                results_matrix.iloc[i, j] = "0 (0.0%)"

    # Display the results matrix
    print(total_test)
    print(results_matrix)


    print("Analyzing booked rates for segment: pg_and_1_plus_sbfe_tradeline_and_fico_no_hit")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'pg_and_1_plus_sbfe_tradeline_and_fico_no_hit')]

    # Define fico score bands and ln_score ranges
    ln_bands = [500, 550, 600, 650, 700, 750, 800, 850, 900]
    ln_ranges = list(zip(ln_bands[:-1], ln_bands[1:]))
    ln_ranges.append((ln_bands[-1], 900))

    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=ln_ranges)
    results_matrix['counts'] = ''
    results_matrix = results_matrix.fillna('')

    # Iterate through fico bands and ln_score ranges
    for i in range(len(ln_ranges)):
        ln_lower = ln_ranges[i][0]
        ln_upper = ln_ranges[i][1]

        filtered_df = data[
            (data['ln_score'] >= ln_lower) &
            (data['ln_score'] < ln_upper)
        ]

        total_rows = len(filtered_df)
        if total_rows > 0:
            booked_rows = filtered_df['booked'].sum()
            booked_rate = (booked_rows / total_rows) * 100
            results_matrix['counts'].iloc[i] = f"{total_rows} ({booked_rate:.1f}%)"
        else:
            results_matrix['counts'].iloc[i] = "0 (0.0%)"

    # Display the results matrix
    print(results_matrix)


    print("Analyzing booked rates for segment: pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_hit')]


    # Define fico score bands and ln_score ranges
    fico_bands = [580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840]
    fico_ranges = list(zip(fico_bands[:-1], fico_bands[1:]))
    fico_ranges.append((fico_bands[-1], 900))
    ln_bands = [500, 550, 600, 650, 700, 750, 800, 850, 900]
    ln_ranges = list(zip(ln_bands[:-1], ln_bands[1:]))
    ln_ranges.append((ln_bands[-1], 900))

    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=fico_ranges, columns=ln_ranges)  # Use ln_ranges directly for columns
    results_matrix = results_matrix.fillna('')

    # Iterate through fico bands and ln_score ranges
    for i in range(len(fico_ranges)):
        for j in range(len(ln_ranges)):  # Iterate through all ln_ranges
            fico_lower = fico_ranges[i][0]
            fico_upper = fico_ranges[i][1]
            ln_lower = ln_ranges[j][0]
            ln_upper = ln_ranges[j][1]

            filtered_df = data[
                (data['fico_score'] >= fico_lower) &
                (data['fico_score'] < fico_upper) &
                (data['ln_score'] >= ln_lower) &
                (data['ln_score'] < ln_upper)
            ]

            total_rows = len(filtered_df)
            if total_rows > 0:
                booked_rows = filtered_df['booked'].sum()
                booked_rate = (booked_rows / total_rows) * 100
                results_matrix.iloc[i, j] = f"{total_rows} ({booked_rate:.1f}%)"
            else:
                results_matrix.iloc[i, j] = "0 (0.0%)"

    # Display the results matrix
    print(results_matrix)


    print("Analyzing booked rates for segment: pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_1_plus_sba_tradeline_and_fico_no_hit')]


    # Define fico score bands and ln_score ranges
    ln_bands = [500, 550, 600, 650, 700, 750, 800, 850, 900]
    ln_ranges = list(zip(ln_bands[:-1], ln_bands[1:]))
    ln_ranges.append((ln_bands[-1], 900))


    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=ln_ranges)
    results_matrix['counts'] = ''
    results_matrix = results_matrix.fillna('')

    # Iterate through fico bands and ln_score ranges
    for i in range(len(ln_ranges)):
        ln_lower = ln_ranges[i][0]
        ln_upper = ln_ranges[i][1]

        filtered_df = data[
            (data['ln_score'] >= ln_lower) &
            (data['ln_score'] < ln_upper)
        ]

        total_rows = len(filtered_df)
        if total_rows > 0:
            booked_rows = filtered_df['booked'].sum()
            booked_rate = (booked_rows / total_rows) * 100
            results_matrix['counts'].iloc[i] = f"{total_rows} ({booked_rate:.1f}%)"
        else:
            results_matrix['counts'].iloc[i] = "0 (0.0%)"

    # Display the results matrix
    print(results_matrix)


    print("Analyzing booked rates for segment: pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'pg_and_no_sbfe_tradeline_and_no_sba_tradeline_and_fico_hit')]


    # Define fico score bands and ln_score ranges
    fico_bands = [580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840]
    fico_ranges = list(zip(fico_bands[:-1], fico_bands[1:]))
    fico_ranges.append((fico_bands[-1], 900))


    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=fico_ranges)
    results_matrix['counts'] = ''
    results_matrix = results_matrix.fillna('')

    # Iterate through fico bands and ln_score ranges
    for i in range(len(fico_ranges)):
        fico_lower = fico_ranges[i][0]
        fico_upper = fico_ranges[i][1]

        filtered_df = data[
            (data['fico_score'] >= fico_lower) &
            (data['fico_score'] < fico_upper)
        ]

        total_rows = len(filtered_df)
        if total_rows > 0:
            booked_rows = filtered_df['booked'].sum()
            booked_rate = (booked_rows / total_rows) * 100
            results_matrix['counts'].iloc[i] = f"{total_rows} ({booked_rate:.1f}%)"
        else:
            results_matrix['counts'].iloc[i] = "0 (0.0%)"

    # Display the results matrix
    print(results_matrix)
    

    print("Analyzing booked rates for segment: no_pg_and_1_plus_sbfe_tradeline")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'no_pg_and_1_plus_sbfe_tradeline')]

    # Define fico score bands and ln_score ranges
    ln_bands = [500, 550, 600, 650, 700, 750, 800, 850, 900]
    ln_ranges = list(zip(ln_bands[:-1], ln_bands[1:]))
    ln_ranges.append((ln_bands[-1], 900))

    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=ln_ranges)
    results_matrix['counts'] = ''
    results_matrix = results_matrix.fillna('')

    # Iterate through fico bands and ln_score ranges
    for i in range(len(ln_ranges)):
        ln_lower = ln_ranges[i][0]
        ln_upper = ln_ranges[i][1]

        filtered_df = data[
            (data['ln_score'] >= ln_lower) &
            (data['ln_score'] < ln_upper)
        ]

        total_rows = len(filtered_df)
        if total_rows > 0:
            booked_rows = filtered_df['booked'].sum()
            booked_rate = (booked_rows / total_rows) * 100
            results_matrix['counts'].iloc[i] = f"{total_rows} ({booked_rate:.1f}%)"
        else:
            results_matrix['counts'].iloc[i] = "0 (0.0%)"

    # Display the results matrix
    print(results_matrix)


    print("Analyzing booked rates for segment: no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline")
    data = data_dev_scored.copy()
    data = data[(data['risk_grade_path'] == 'no_pg_and_no_1_plus_sbfe_tradeline_and_1_plus_sba_tradeline')]

    # Define fico score bands and ln_score ranges
    ln_bands = [500, 550, 600, 650, 700, 750, 800, 850, 900]
    ln_ranges = list(zip(ln_bands[:-1], ln_bands[1:]))
    ln_ranges.append((ln_bands[-1], 900))


    # Create an empty matrix to store results
    results_matrix = pd.DataFrame(index=ln_ranges)
    results_matrix['counts'] = ''
    results_matrix = results_matrix.fillna('')

    # Iterate through fico bands and ln_score ranges
    for i in range(len(ln_ranges)):
        ln_lower = ln_ranges[i][0]
        ln_upper = ln_ranges[i][1]

        filtered_df = data[
            (data['ln_score'] >= ln_lower) &
            (data['ln_score'] < ln_upper)
        ]

        total_rows = len(filtered_df)
        if total_rows > 0:
            booked_rows = filtered_df['booked'].sum()
            booked_rate = (booked_rows / total_rows) * 100
            results_matrix['counts'].iloc[i] = f"{total_rows} ({booked_rate:.1f}%)"
        else:
            results_matrix['counts'].iloc[i] = "0 (0.0%)"

    # Display the results matrix
    print(results_matrix)