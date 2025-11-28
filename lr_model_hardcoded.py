"""
Hardcoded Logistic Regression Model for PRE vs POS Selection
Auto-generated from model_evaluation.py
"""
import numpy as np

# Model trained on features: ['k', 'num_survivors', 'selectivity']
# Feature order: k, num_survivors, selectivity

# Scaler parameters (StandardScaler)
SCALER_MEAN = np.array([2687.2571942446, 81211.6438848921, 0.5414109592])
SCALER_SCALE = np.array([3341.8898775446, 35212.3631437106, 0.2347490876])

# Logistic Regression parameters
LR_COEF = np.array([-1.3276164995, 2.2549849548, 2.2549849548])
LR_INTERCEPT = 9.9547678062


def predict_pos_faster(k: int, num_survivors: int, total_docs: int = 150000) -> bool:
    """
    Predict whether POS search will be faster than PRE search.

    Args:
        k: Number of results requested
        num_survivors: Estimated number of survivors after predicates
        total_docs: Total number of documents (default: 150000)

    Returns:
        True if POS is predicted to be faster, False if PRE is predicted to be faster
    """
    # Calculate selectivity
    selectivity = num_survivors / total_docs

    # Create feature vector
    features = np.array([k, num_survivors, selectivity])

    # Scale features
    features_scaled = (features - SCALER_MEAN) / SCALER_SCALE

    # Calculate logistic regression prediction
    logit = np.dot(features_scaled, LR_COEF) + LR_INTERCEPT

    # Apply sigmoid and get prediction
    probability_pos = 1 / (1 + np.exp(-logit))

    # Predict POS if probability > 0.5
    return probability_pos > 0.5


def get_prediction_probability(k: int, num_survivors: int, total_docs: int = 150000) -> tuple:
    """
    Get the probability of POS being faster.

    Args:
        k: Number of results requested
        num_survivors: Estimated number of survivors after predicates
        total_docs: Total number of documents (default: 150000)

    Returns:
        Tuple of (prediction, prob_pre, prob_pos)
    """
    # Calculate selectivity
    selectivity = num_survivors / total_docs

    # Create feature vector
    features = np.array([k, num_survivors, selectivity])

    # Scale features
    features_scaled = (features - SCALER_MEAN) / SCALER_SCALE

    # Calculate logistic regression prediction
    logit = np.dot(features_scaled, LR_COEF) + LR_INTERCEPT

    # Apply sigmoid
    probability_pos = 1 / (1 + np.exp(-logit))
    probability_pre = 1 - probability_pos

    prediction = "POS" if probability_pos > 0.5 else "PRE"

    return prediction, probability_pre, probability_pos
