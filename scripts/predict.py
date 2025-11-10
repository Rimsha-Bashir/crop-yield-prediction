
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


eta = 0.1
max_depth = 7
min_child_weight = 5 

model_file = '../model/xgboost_eta=%s_depth=%s_minchild=%s_round=200.bin'%(eta, max_depth, min_child_weight)

with open(model_file, 'rb') as f_in:
    model, dv = pickle.load(f_in)

def predict(df):

    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    features = dv.get_feature_names_out().tolist()
    d = xgb.DMatrix(X, feature_names=features)

    y_pred_log = model.predict(d)
    y_pred = np.expm1(y_pred_log)  # inverse log transform

    return y_pred


if __name__ == '__main__':

    example = pd.DataFrame([{
        'country': 'albania',
        'crop': 'soybeans',
        'year': 2016,
        'average_rain_fall_mm_per_year': 1440.0,
        'pesticide_tonnes': 1419,
        'avg_temp': 17
    }])
    
    preds = predict(example)
    print(f"Predicted yield: {preds[0]:.2f}")
