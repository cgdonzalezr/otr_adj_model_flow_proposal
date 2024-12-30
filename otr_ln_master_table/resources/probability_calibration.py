# https://drive.google.com/file/d/1k03EXcCVuKOKcym8DPR8C_qOUTQyf86x/view?usp=drive_link 

def correct_probabilities(ps, beta):
    """Correct biased probabilities from undersampled data.

    Args:
        ps (array-like): Array of biased probabilities (output by the model trained on undersampled data).
        beta (float): Undersampling rate (proportion of negative class instances retained during training).

    Returns:
        array-like: Corrected probabilities that align with the true distribution of the original dataset.
    """
    return (beta * ps) / ((beta * ps) - ps + 1)

def correct_threshold(beta, tau_s):
    """Adjust threshold for corrected probabilities to maintain accuracy.

    Args:
        beta (float): Undersampling rate.
        tau_s (float): Threshold used in the undersampled dataset.

    Returns:
        float: Corrected threshold that works with the unbiased probabilities to classify samples accurately.
    """
    return (beta * tau_s) / ((beta - 1) * tau_s + 1)

# Example Parameters
beta = 0.111  # Undersampling rate (for example 1,000 negatives retained out of 9,000)
tau_s = 0.5   # Default threshold to be changed

# Model output for test data (biased probabilities from undersampled training)
# Assumes the model is already trained, and predict_proba() gives probabilities for the positive class.
# biased_probabilities = model.predict_proba(X_test)[:, 1]

# Correct probabilities using the correction formula
# corrected_probabilities = correct_probabilities(biased_probabilities, beta)

# Adjust the threshold for corrected probabilities
# corrected_threshold = correct_threshold(beta, tau_s)

# Classify transactions based on corrected probabilities and adjusted threshold
# predictions = (corrected_probabilities > corrected_threshold).astype(int)