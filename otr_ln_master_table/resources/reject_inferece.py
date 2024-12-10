import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def augmentation(xf, xnf, yf, use_speedglm=False):  # Removed speedglm option as it's R-specific
    """
    Augmentation (re-weighting) method for reject inference.

    Args:
        xf (pd.DataFrame or np.ndarray): Features of financed applicants.
        xnf (pd.DataFrame or np.ndarray): Features of not financed applicants.
        yf (pd.Series or np.ndarray): Target variable (repayment status) of financed applicants.
        use_speedglm: No longer applicable. Kept for compatibility reasons.
    
    Returns:
        dict: A dictionary containing the trained models.
    """
    # Combine financed and not financed data
    x_combined = pd.concat([pd.DataFrame(xf), pd.DataFrame(xnf)], axis=0)
    # Create a financing indicator (1 for financed, 0 for not financed)
    z = np.concatenate([np.ones(len(xf)), np.zeros(len(xnf))])

    # Train a model to predict financing status (acceptance model)
    acceptance_model = LogisticRegression(solver='liblinear').fit(x_combined, z)

    # Calculate weights for financed applicants
    acceptance_probs = acceptance_model.predict_proba(xf)[:, 1]
    weights = 1 / acceptance_probs
    
    # Train a weighted model on financed applicants (infered model)
    infered_model = LogisticRegression(solver='liblinear').fit(xf, yf, sample_weight=weights)
    
    # Train a model on only financed data (financed_model)
    financed_model = LogisticRegression(solver='liblinear').fit(xf, yf)

    return {
        'method_name': 'augmentation',
        'financed_model': financed_model,
        'acceptance_model': acceptance_model,
        'infered_model': infered_model
    }

def fuzzy_augmentation(xf, xnf, yf, use_speedglm=False): # Removed speedglm option as it's R-specific
    """
    Fuzzy augmentation method for reject inference.

    Args:
        xf (pd.DataFrame or np.ndarray): Features of financed applicants.
        xnf (pd.DataFrame or np.ndarray): Features of not financed applicants.
        yf (pd.Series or np.ndarray): Target variable (repayment status) of financed applicants.
        use_speedglm: No longer applicable.

    Returns:
         dict: A dictionary containing the trained models.
    """

    # Train initial model on financed applicants
    financed_model = LogisticRegression(solver='liblinear').fit(xf, yf)

    # Predict probabilities for not financed applicants
    y_prob = financed_model.predict_proba(xnf)[:, 1]

    # Create "fuzzy" target variable for rejected applicants
    y_fuzzy = np.where(y_prob >= 0.5, 1, 0) # Simpler fuzzy augmentation

    # Combine data and retrain
    x_combined = np.concatenate([xf, xnf], axis=0)
    y_combined = np.concatenate([yf, y_fuzzy], axis=0)

    infered_model = LogisticRegression(solver='liblinear').fit(x_combined, y_combined)

    return {
        'method_name': 'fuzzy_augmentation',
        'financed_model': financed_model,
        'acceptance_model': None, # No acceptance model in this approach
        'infered_model': infered_model
    }



def reclassification(xf, xnf, yf, use_speedglm=False): # Removed speedglm option as it's R-specific
    """
    Reclassification method for reject inference.

    Args:
        xf (pd.DataFrame or np.ndarray): Features of financed applicants.
        xnf (pd.DataFrame or np.ndarray): Features of not financed applicants.
        yf (pd.Series or np.ndarray): Target variable (repayment status) of financed applicants.
        use_speedglm: No longer applicable.

    Returns:
        dict: A dictionary containing the trained models.
    """

    # Train initial model on financed applicants
    financed_model = LogisticRegression(solver='liblinear').fit(xf, yf)

    # Classify not financed applicants
    y_pred = financed_model.predict(xnf)

    # Combine data and retrain
    x_combined = np.concatenate([xf, xnf], axis=0)
    y_combined = np.concatenate([yf, y_pred], axis=0)

    infered_model = LogisticRegression(solver='liblinear').fit(x_combined, y_combined)

    return {
        'method_name': 'reclassification',
        'financed_model': financed_model,
        'acceptance_model': None, # No acceptance model used here
        'infered_model': infered_model
    }





def twins(xf, xnf, yf, use_speedglm=False): # Removed speedglm option as it's R-specific
    """
    Twins method for reject inference.

    Args:
        xf (pd.DataFrame or np.ndarray): Features of financed applicants.
        xnf (pd.DataFrame or np.ndarray): Features of not financed applicants.
        yf (pd.Series or np.ndarray): Target variable (repayment status) of financed applicants.
        use_speedglm: No longer applicable.

    Returns:
         dict: A dictionary containing the trained models.
    """
    x_combined = np.concatenate([xf, xnf], axis=0)
    z = np.concatenate([np.ones(len(xf)), np.zeros(len(xnf))])

    # Train a model to predict financing status (acceptance model)
    acceptance_model = LogisticRegression(solver='liblinear').fit(x_combined, z)

    # Train a model on financed applicants
    financed_model = LogisticRegression(solver='liblinear').fit(xf, yf)
    
    # Create new features from model predictions
    xf_new_features =  np.column_stack([financed_model.predict_proba(xf)[:,1], acceptance_model.predict_proba(xf)[:,1]])
    
    # Train a model with the new features
    infered_model = LogisticRegression(solver='liblinear').fit(xf_new_features, yf)

    return {
        'method_name': 'twins',
        'financed_model': financed_model,
        'acceptance_model': acceptance_model,
        'infered_model': infered_model
    }






def parcelling(xf, xnf, yf, n_bins=10, prudence_factors=None, use_speedglm=False):  # Removed speedglm option, simplified prudence factors
    """
    Parcelling method for reject inference.

    Args:
        xf (pd.DataFrame or np.ndarray): Features of financed applicants.
        xnf (pd.DataFrame or np.ndarray): Features of not financed applicants.
        yf (pd.Series or np.ndarray): Target variable (repayment status) of financed applicants.
        n_bins (int): Number of bins for score bands.
        prudence_factors (list or None): Prudence factors for each bin. If None, defaults to increasing sequence.
        use_speedglm: No longer applicable.

    Returns:
        dict: A dictionary containing the trained models.

    """
    # Train initial model on financed applicants
    financed_model = LogisticRegression(solver='liblinear').fit(xf, yf)
    
    # Predict probabilities for all applicants
    scores_financed = financed_model.predict_proba(xf)[:, 1]
    scores_not_financed = financed_model.predict_proba(xnf)[:, 1]
    
    # Create score bands
    _, bins = pd.qcut(scores_financed, q=n_bins, retbins=True, duplicates='drop')
    
    if prudence_factors is None:
        prudence_factors = list(range(1, len(bins))) # Default: simple increasing sequence
        
    # Apply prudence factors and combine data
    yf_parcelling = yf.copy()

    for i in range(len(bins) -1 ):
        indices = (scores_not_financed > bins[i]) & (scores_not_financed <= bins[i+1])
        # No changes needed for financed, just applying prudence factors to weights below
       

    # Create weights based on score bands for financed data
    weights = np.ones_like(yf_parcelling, dtype=float)

    for i in range(len(bins) - 1):
        indices_financed = (scores_financed > bins[i]) & (scores_financed <= bins[i + 1])
        weights[indices_financed] = prudence_factors[i]

    infered_model = LogisticRegression(solver='liblinear').fit(xf, yf_parcelling, sample_weight=weights)

    return {
        'method_name': 'parcelling',
        'financed_model': financed_model,
        'acceptance_model': None, # No acceptance model directly used
        'infered_model': infered_model
    }