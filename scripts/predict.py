import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import os

# -------------------------------
# Load model and DictVectorizer
# -------------------------------
# Use os.path.join for better cross-platform path handling
BASE_DIR = os.path.dirname(__file__)  # directory of predict.py
MODEL_DIR = os.path.join(BASE_DIR, '../model')
MODEL_FILE = "xgboost_eta=0.1_depth=7_minchild=5_round=200.bin"
MODEL_PATH = os.path.abspath(os.path.join(MODEL_DIR, MODEL_FILE))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(MODEL_PATH, 'rb') as f_in:
    model, dv = pickle.load(f_in)

"""def predict(df):

    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    features = dv.get_feature_names_out().tolist()
    d = xgb.DMatrix(X, feature_names=features)

    y_pred_log = model.predict(d)
    y_pred = np.expm1(y_pred_log)  # inverse log transform

    return y_pred
"""


def predict(data):
    """
    Runs a prediction on either a single JSON dict or a pandas DataFrame.

    Parameters:
        data: dict or pd.DataFrame
            Example dict: {
                'country': 'albania',
                'crop': 'soybeans',
                'year': 2016,
                'average_rain_fall_mm_per_year': 1440.0,
                'pesticide_tonnes': 1419,
                'avg_temp': 17
            }

    Returns:
        float: predicted crop yield
    """
    
    # Handle JSON input (dict)
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Input must be a dictionary or a pandas DataFrame.")
    
    # Convert to model format
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    features = dv.get_feature_names_out().tolist()
    d = xgb.DMatrix(X, feature_names=features)

    # Predict and inverse log-transform
    y_pred_log = model.predict(d)
    y_pred = np.expm1(y_pred_log)

    return float(y_pred[0])  # return scalar for JSON serialization



if __name__ == '__main__':

    example = {
        'country': 'albania',
        'crop': 'soybeans',
        'year': 2016,
        'average_rain_fall_mm_per_year': 1440.0,
        'pesticide_tonnes': 1419,
        'avg_temp': 17
    }
    
    preds = predict(example)
    print(f"Predicted yield: {preds[0]:.2f}")
