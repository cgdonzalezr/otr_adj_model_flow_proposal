import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from patsy import dmatrix, dmatrices

import warnings
warnings.filterwarnings(
    action='ignore',
    category=UserWarning,
    module='snowflake.connector'
)

# https://medium.com/@hjdlopes/should-we-reject-reject-inference-an-empirical-study-4f1e5d86bcf4
# https://cer.business-school.ed.ac.uk/wp-content/uploads/sites/55/2017/03/Paper-25-Presentation.pdf
# https://www.mathworks.com/help/risk/reject-inference-for-credit-scorecards.html#RejectingInferenceMethodologyExample-6
# https://docs.google.com/presentation/d/1Mtw9ATppHdINf-Vq6-rYALeofAecgLoAekxgOtl4nxc/edit#slide=id.g30a66dcb834_0_1
# https://www.crc.business-school.ed.ac.uk/sites/crc/files/2023-10/Modified-logistic-regression-using-the-EM-algorithm-for-reject-inference.pdf

# https://docs.google.com/presentation/d/1ZH63C0UjLsQimlhcEzbjB1rSv9um_MX_UzMZ0z457b8/edit#slide=id.g31e1a061c2e_0_15463

import os
import sys

current_dir = os.getcwd()
resources_folder = os.path.join(current_dir, '..', 'resources')
sys.path.append(resources_folder)

from snowflake_connector import conn
with open("../resources/sql/OTR_ADJ_RISK_PREPROCESSED_SAMPLE_MODEL_MONITORING.sql") as nam_file:
    nam_sql = nam_file.read()

cur = conn.cursor()

cur.execute(nam_sql)

df = cur.fetch_pandas_all()
df = pd.DataFrame.from_records(iter(cur), columns=[x[0] for x in cur.description])

cur.close()



def hard_cutoff_augmentation(xf, xnf, yf):
    """
    Implements the Hard-Cutoff augmentation technique for reject inference.

    Args:
        xf (pd.DataFrame): Features of accepted applicants.
        xnf (pd.DataFrame): Features of rejected applicants.
        yf (pd.Series): Target variable (good/bad status) of accepted applicants.
                      Assumes 0 for good, 1 for bad.

    Returns:
        dict: A dictionary containing the trained Logistic Regression models and method name:
               - 'method_name': The name of the method.
               - 'financed_model': The model trained on accepts only.
               - 'acceptance_model': None (not applicable for this method).
               - 'infered_model': The model trained on the combined data after hard-cutoff augmentation.
    """
    print("Hard-Cutoff augmentation method")

    # Ensure yf is numeric (0 for good, 1 for bad)
    yf_numeric = pd.Series(np.where(yf == 0, 0, 1), index=yf.index)

    # Calculate bad_rate_accepts for observed pop. Used to assign default or not
    bad_rate_accepts = yf_numeric.mean()

    # 1. Build a scorecard using the accepts only
    print("Training model on accepts only...")
    model_accepts_only = Logit(yf_numeric, xf).fit()
    print(model_accepts_only.summary())

    # 2. Score the rejects
    print("Scoring rejects...")
    scores_rejects = model_accepts_only.predict(xnf)

    # 3. Choose a Bad Rate and Classify Rejects
    print(f"Classifying rejects using accepts bad rate: {bad_rate_accepts}, having {len(yf_numeric)} accepts")

    # Assuming 1 represents 'bad'
    # If the score is higher than the cutoff, we classify it as bad (1)
    response_rejects = pd.Series(np.where(scores_rejects > bad_rate_accepts, 1, 0), index=xnf.index)
    
    # Count of estimated bads
    print(f"Estimated bads: {response_rejects.sum()}")
    print(f"Estimated goods: {len(response_rejects) - response_rejects.sum()}")
    
    # 4. Combine Accepts and Rejects into a New Data Set
    print("Creating combined dataset...")
    accepts_data = xf.assign(status=yf_numeric)
    rejects_data = xnf.assign(status=response_rejects)
    combined_data = pd.concat([accepts_data, rejects_data])

    # 5. Fit a logistic regression model for the combined data
    print("Training model on combined data with hard-cutoff augmentation...")
    model_combined = Logit(combined_data['status'], combined_data[xf.columns]).fit() # Use original feature columns
    print(model_combined.summary())

    return {
        'method_name': 'hard_cutoff_augmentation',
        'financed_model': model_accepts_only,
        'acceptance_model': None,
        'infered_model': model_combined
    }





def fuzzy_augmentation(xf, xnf, yf):
    """
    Implements the Fuzzy Augmentation technique for reject inference.

    Args:
        xf (pd.DataFrame): Features of accepted applicants.
        xnf (pd.DataFrame): Features of rejected applicants.
        yf (pd.Series): Target variable (good/bad status) of accepted applicants.

    Returns:
        tuple: A tuple containing two trained Logistic Regression models:
               - model_accepts_only: The model trained on accepts only.
               - model_combined: The model trained on the combined data with fuzzy augmentation.
    """
    print("Fuzzy augmentation method")

    # Ensure yf is numeric (0 for good, 1 for bad - assuming this from the example)
    yf_numeric = pd.Series(np.where(yf == 0, 0, 1), index=yf.index)

    # 1. Build a scorecard using the accepts only
    print("Training model on accepts only...")
    model_accepts_only = Logit(yf_numeric, xf).fit()
    print(model_accepts_only.summary())

    # 2. Score the rejects
    print("Scoring rejects...")
    pred_prob_rejects = model_accepts_only.predict(xnf)

    # 3. Create the combined data set with weighted rejects
    print("Creating combined dataset with weighted rejects...")
    n_rejects = len(xnf)
    combined_data = pd.DataFrame()

    # Add accepts data
    combined_data = pd.concat([combined_data, xf.assign(status=yf_numeric, Weights=1)])

    # Add fuzzy augmented rejects
    rejects_good = xnf.copy()
    rejects_good['status'] = 0  # 0 is the good label
    rejects_good['Weights'] = 1 - pred_prob_rejects

    rejects_bad = xnf.copy()
    rejects_bad['status'] = 1   # 1 is default/bad
    rejects_bad['Weights'] = pred_prob_rejects

    combined_data = pd.concat([combined_data, rejects_good, rejects_bad])

    # 4. Create and fit a scorecard on the combined data
    print("Training model on combined data with fuzzy augmentation...")
    model_combined = Logit(combined_data['status'], combined_data, weights=combined_data['Weights']).fit(maxiter=100)
    print(model_combined.summary())

    return {
        'method_name': 'fuzzy_augmentation',
        'financed_model': model_accepts_only,  
        'acceptance_model': None,
        'infered_model': model_combined
    }




# def em_algorithm(xf, xnf, yf, max_iterations=100, convergence_threshold=1e-5):
#     """
#     Implements the Expectation-Maximization (EM) algorithm for reject inference.

#     Args:
#         xf (pd.DataFrame): Features of accepted applicants.
#         xnf (pd.DataFrame): Features of rejected applicants.
#         yf (pd.Series): Target variable (good/bad status) of accepted applicants.
#                       Assumes 0 for good, 1 for bad.
#         max_iterations (int): Maximum number of EM iterations.
#         convergence_threshold (float): Threshold for convergence of model parameters.

#     Returns:
#         dict: A dictionary containing the trained Logistic Regression models and method name:
#                - 'method_name': The name of the method.
#                - 'financed_model': The model trained on accepts only.
#                - 'acceptance_model': None (not applicable for this method).
#                - 'infered_model': The model trained using the EM algorithm.
#     """
#     print("Expectation Maximization (EM) algorithm")

#     # Num of accepts
#     print(f"Number of accepts: {len(yf)}")

#     # Num of rejects
#     print(f"Number of rejects: {len(xnf)}")

#     # Ensure yf is numeric (0 for good, 1 for bad)
#     yf_numeric = pd.Series(np.where(yf == 0, 0, 1), index=yf.index)

#     # Calculate the bad rate of the accepts
#     # bad_rate_accepts = yf_numeric.mean()
#     bad_rate_accepts = 0.03

#     # 1. Initialization: Train a model with the accepts only
#     print("Initialization: Training model on accepts only...")
#     model_accepts_only = Logit(yf_numeric, xf).fit()
#     print(model_accepts_only.summary())

#     current_model = model_accepts_only
#     previous_params = None

#     for iteration in range(max_iterations):
#         print(f"EM Iteration: {iteration + 1}")

#         # E-step: Calculate the probability of being bad for the rejects and assign labels
#         print("E-step: Calculating probabilities and assigning labels to the rejects...")
#         scores_rejects = current_model.predict(xnf)

#         # option 1 - Hard cutoff (assigns 1 if score > bad_rate_accepts) (Coefficents explode to reject more)
#         print(f"Classifying rejects using accepts bad rate: {bad_rate_accepts}, having {len(yf_numeric)} accepts")
#         response_rejects = pd.Series(np.where(scores_rejects > bad_rate_accepts, 1, 0), index=xnf.index)
#         # Count of estimated bads
#         print(f"Estimated bads: {response_rejects.sum()}")
#         print(f"Estimated goods: {len(response_rejects) - response_rejects.sum()}")
          
#         # option 2 - Rejected target variable is the probability of being bad (Coefficents not changing)
#         # response_rejects = scores_rejects

#         # M-step: Create the combined dataset and train a new model
#         print("M-step: Creating combined dataset and training the model...")
#         accepts_data = xf.assign(status=yf_numeric)
#         rejects_data = xnf.assign(status=response_rejects)
#         combined_data = pd.concat([accepts_data, rejects_data])

#         em_model = Logit(combined_data['status'], combined_data[xf.columns]).fit(maxiter=100, disp=False)
#         print(em_model.summary())

#         # Check for convergence
#         if previous_params is not None:
#             param_change = np.sum(np.abs(em_model.params - previous_params))
#             print(f"Change in parameters: {param_change}")
#             if param_change < convergence_threshold:
#                 print("Convergence reached.")
#                 break

#         previous_params = em_model.params.copy()
#         current_model = em_model

#     return {
#         'method_name': 'em_algorithm',
#         'financed_model': model_accepts_only,
#         'acceptance_model': None,
#         'infered_model': current_model
#     }





def em_algorithm(xf, xnf, yf, max_iterations=100, convergence_threshold=1e-5):
    """
    Implements the Expectation-Maximization (EM) algorithm for reject inference using statsmodels GLM.

    Args:
        xf (pd.DataFrame): Features of accepted applicants.
        xnf (pd.DataFrame): Features of rejected applicants.
        yf (pd.Series): Target variable (good/bad status) for accepted applicants.
                        Assumes 0 for good and 1 for bad.
        max_iterations (int): Maximum number of EM iterations.
        convergence_threshold (float): Convergence threshold for model parameters.

    Returns:
        dict: A dictionary containing the trained models and method name:
              - 'method_name': Name of the method.
              - 'financed_model': Model trained on accepted applicants only.
              - 'acceptance_model': None (not applicable for this method).
              - 'infered_model': Model trained using the EM algorithm.
    """
    # Debug: Check initial data
    print(f"Number of accepts: {len(yf)}")
    print(f"Number of rejects: {len(xnf)}")
    print(f"Accepted feature columns: {list(xf.columns)}")
    print(f"Rejected feature columns: {list(xnf.columns)}")

    # Convert inputs to numpy arrays
    X_accepted = xf.values if isinstance(xf, pd.DataFrame) else xf
    X_rejected = xnf.values if isinstance(xnf, pd.DataFrame) else xnf
    y_accepted = yf.values if isinstance(yf, pd.Series) else yf

    # Debug: Ensure shapes are consistent
    print(f"X_accepted shape: {X_accepted.shape}, X_rejected shape: {X_rejected.shape}, y_accepted shape: {y_accepted.shape}")

    # Train the financed model on accepted applicants only using GLM (logistic regression)
    print("Training initial model on accepted applicants...")
    financed_model = sm.GLM(y_accepted, X_accepted, family=sm.families.Binomial()).fit()
    print("Initial financed model parameters:")
    print(financed_model.params)
    
    # Initialize the EM model with the financed model
    model = financed_model
    prev_params = None

    for i in range(max_iterations):
        # === E-Step ===
        # Predict probabilities for rejected applicants
        if X_rejected.shape[0] > 0:
            p_rejected = model.predict(X_rejected)
        else:
            print("No rejected applicants to process. Exiting loop.")
            break

        # === M-Step ===
        # Create the augmented dataset:
        X_augmented = np.vstack([X_accepted, X_rejected, X_rejected])
        y_augmented = np.concatenate([y_accepted, np.ones(len(X_rejected)), np.zeros(len(X_rejected))])
        sample_weights = np.concatenate([np.ones(len(X_accepted)), p_rejected, 1 - p_rejected])

        # Fit a new GLM model on the augmented data with frequency weights
        new_model = sm.GLM(y_augmented, X_augmented, family=sm.families.Binomial(), freq_weights=sample_weights).fit()

        # Compute parameter change for convergence check
        if prev_params is not None:
            param_change = np.max(np.abs(new_model.params - prev_params))
            print(f"Iteration {i+1} parameter change: {param_change:.6f}")
            if param_change < convergence_threshold:
                model = new_model
                print("Convergence reached.")
                break
        else:
            print(f"Iteration {i+1} initial parameters:")
            print(new_model.params)

        prev_params = new_model.params.copy()
        model = new_model

    print("Final inferred model parameters:")
    print(model.params)
    
    return {
        'method_name': 'EM Reject Inference (statsmodels)',
        'financed_model': financed_model,
        'acceptance_model': None,
        'infered_model': model
    }

