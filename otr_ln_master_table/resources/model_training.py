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



def generate_model_parameters_df(models_dict: Dict[str, BinaryResultsWrapper]) -> pd.DataFrame:
    """Generate a dataframe that holds the parameters for the score blending."""

    df_list = []
    # get the logistic regression parameters for all segments
    for segment, model in models_dict.items():
        df = model.params.to_frame().transpose()
        df["segment"] = segment
        df_list.append(df)

    # combine results
    df_params = pd.concat(df_list, sort=False)

    # fix format
    df_params = df_params[["segment"] + [x for x in df_params.columns if x != "segment"]]
    df_params = df_params.reset_index(drop=True)

    return df_params
    

def fit_risk_grades(
    segment_name: str,
    data: pd.DataFrame,
    model_cols: List[str],
    core_model_vars: List[str],
    train_filter: pd.Series = None,
    NORMALIZE_SCORES: bool = True,
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

    # Fit logistic regression
    logistic_regressor = Logit(
        endog=data.loc[train_filter, "is_bad"].astype("bool"),
        exog=data.loc[train_filter, model_cols_norm],
    ).fit(maxiter=100, method="bfgs")


    return logistic_regressor, variable_normalizations





def fit_models_for_segments(data: pd.DataFrame, RISK_GRADE_SEGMENTS, NORMALIZE_SCORES) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict]:
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
        )

        # store results
        models_dict[segment_name] = model_i
        normalization_dict[segment_name] = variable_normalizations_i

    # combine results

    return  models_dict, normalization_dict


def generate_model_parameters_df(models_dict: Dict[str, BinaryResultsWrapper]) -> pd.DataFrame:
    """Generate a dataframe that holds the parameters for the score blending."""

    df_list = []
    # get the logistic regression parameters for all segments
    for segment, model in models_dict.items():
        df = model.params.to_frame().transpose()
        df["segment"] = segment
        df_list.append(df)

    # combine results
    df_params = pd.concat(df_list, sort=False)

    # fix format
    df_params = df_params[["segment"] + [x for x in df_params.columns if x != "segment"]]
    df_params = df_params.reset_index(drop=True)

    return df_params



def apply_fitted_models_to_data(
    data: pd.DataFrame,
    models_dict: Dict,
    normalization_dict: Dict,
    RISK_GRADE_SEGMENTS: Dict,
    NORMALIZE_SCORES: bool = True,
) -> pd.DataFrame:
    """Apply fitted logistic regression models to data.

    Parameters
    ----------
    data
        Data to apply models to
    models_dict
        Dictionary of fitted models for each segment
    normalization_dict
        Dictionary of normalization parameters for each segment
    RISK_GRADE_SEGMENTS
        Configuration for segments

    Returns
    -------
    data
        Copy of original data, with model predictions for each segment
    """

    # Avoid changes outside function scope
    data = data.copy()
    
    # iterate over all segments
    for segment_name, params in RISK_GRADE_SEGMENTS.items():
        # filter data for segment
        data_i = filter_df_for_condition(df=data, cond_dict=params["condition"])

        # normalize independent variables if necessary
        if NORMALIZE_SCORES:
            model_cols_norm = []
            for col in params["model_cols"]:
                if (
                    normalization_dict[segment_name] is not None
                    and col in normalization_dict[segment_name]
                ):
                    data_i[col + "_z"] = (
                        data_i[col] - normalization_dict[segment_name][col]["mean"]
                    ) / normalization_dict[segment_name][col]["std"]
                    model_cols_norm.append(col + "_z")
                else:
                    model_cols_norm.append(col)
        else:
            model_cols_norm = params["model_cols"]

        # apply model and store prediction
        if len(data_i) > 0:  # avoid error if segment is empty
            data_i[f"pd_{segment_name}"] = models_dict[segment_name].predict(
                exog=data_i[model_cols_norm]
            )

        # combine results for all segments
        if segment_name == list(RISK_GRADE_SEGMENTS.keys())[0]:
            data_out = data_i
        else:
            data_out = pd.concat([data_out, data_i], axis=0)

    return data_out



# Calculate segment statistics with approval/decline thresholds
def calculate_segment_statistics_with_approval(
    data: pd.DataFrame,
    RISK_GRADE_SEGMENTS: Dict,
    risk_grade_thresholds: Dict[str, Tuple[float, float]],
    approval_thresholds: Dict[str, bool],
) -> pd.DataFrame:
    """Calculate segment statistics with approval/decline decisions."""

    segment_stats = []

    for segment_name, params in RISK_GRADE_SEGMENTS.items():
        # filter data for segment
        data_i = filter_df_for_condition(df=data, cond_dict=params["condition"])

        # Calculate segment statistics
        total_rows = len(data_i)
        num_booked = data_i["booked"].sum()
        num_non_booked = total_rows - num_booked
        booked_is_bad = data_i[data_i["booked"] == 1]["is_bad"].sum()
        predicted_booked = 0
        predicted_booked_is_bad = 0
        predicted_declined = 0

        for risk_grade, (lower_bound, upper_bound) in risk_grade_thresholds.items():
            # Count predicted booked and declined based on risk grade and approval threshold
            mask = (data_i[f"pd_{segment_name}"] >= lower_bound) & (data_i[f"pd_{segment_name}"] < upper_bound)
            if approval_thresholds[risk_grade]:
                predicted_booked += mask.sum()
                predicted_booked_is_bad += data_i[mask]["is_bad"].sum()
            else:
                predicted_declined += mask.sum()

        segment_stats.append(
            {
                "segment_name": segment_name,
                "total_rows": total_rows,
                "num_booked": num_booked,
                "num_non_booked": num_non_booked,
                "booked_is_bad": booked_is_bad,
                "predicted_booked": predicted_booked,
                "predicted_booked_is_bad": predicted_booked_is_bad,
                "predicted_declined": predicted_declined,
            }
        )

    segment_stats_df = pd.DataFrame(segment_stats)
    return segment_stats_df


# Create a function to apply the conditions and choices to a dataframe
def apply_risk_grade_path(
        df: pd.DataFrame,
        conditions: List,
        choices: List,
        ):
  """Applies risk grade path logic to a given dataframe."""
  evaluated_conditions = [condition(df) for condition in conditions]
  df['risk_grade_path'] = np.select(evaluated_conditions, choices, default=np.nan)
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
    binded_df = binded_df[binded_df['risk_grade_path'] == segment]

    # Calculate metrics
    binded_df = binded_df.sort_values(y_pred_proba, ascending = False)
    binded_df["DECILE"] = pd.qcut(binded_df[y_pred_proba].rank(method='first'), q = num_bins, labels = list(range(num_bins, 0, -1)))
    binded_df["TOTAL_EVENT_RATE"] =  binded_df[target].mean()

    # Calculate the number of accounts and default accounts for each decile
    decile_df = binded_df.groupby("DECILE").agg(
        COUNT=(y_pred_proba, "count"),
        DEFAULT=(target, "sum"),
        TOTAL_EVENT_RATE=("TOTAL_EVENT_RATE", "mean"),
    ).reset_index()

    # Calculate the predicted probability range for each decile
    decile_df["PROB_RANGE"] = binded_df.groupby("DECILE")[y_pred_proba].agg(["min", "max"]).apply(lambda x: f"({x['min']:.4f} - {x['max']:.4f}]", axis=1).values
    decile_df["AVG_PROB"] = binded_df.groupby("DECILE")[y_pred_proba].agg(["mean"]).values
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
    plt2.set_ylim(ymin=0, ymax=decile_df["DEFAULT_RATE"].max() + 0.05)
    plt2.set_yticks(np.arange(0, decile_df["DEFAULT_RATE"].max() + 0.05, step=0.05))
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
    risk_grade_df = binded_df.groupby(risk_grade_column).agg(
        COUNT=(y_pred_proba, "count"),
        DEFAULT=(target, "sum"),
        TOTAL_EVENT_RATE=("TOTAL_EVENT_RATE", "mean"),
    ).reset_index()

    # Calculate the avg predicted probability
    risk_grade_df["AVG_PROB"] = binded_df.groupby(risk_grade_column)[y_pred_proba].agg(["mean"]).values
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