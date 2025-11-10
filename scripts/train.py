import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


df = pd.read_csv('../data/yield_final.csv')



categorical = ['country','crop', 'year']

numerical = ['average_rain_fall_mm_per_year',
 'pesticide_tonnes',
 'avg_temp']

eta = 0.1
max_depth = 7
min_child_weight = 5 


## Splitting dataset into train, test and validation

print("Splitting Dataset... \n\n")

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train=df_train.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)
df_val=df_val.reset_index(drop=True)

y_full_train = df_full_train['yield_hg_ha'].values
y_train = df_train.yield_hg_ha.values
y_test = df_test.yield_hg_ha.values
y_val = df_val.yield_hg_ha.values

y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_full_train_log = np.log1p(y_full_train)


del df_full_train['yield_hg_ha']
del df_train['yield_hg_ha']
del df_test['yield_hg_ha']
del df_val['yield_hg_ha']


## Feature Engineering

print("Performing Feature Engineering... \n\n")

# Log-transform highly skewed numerical features
skewed_features = ['pesticide_tonnes', 'average_rain_fall_mm_per_year']
for col in skewed_features:
    df_full_train[col] = np.log1p(df_full_train[col])
    df_train[col] = np.log1p(df_train[col])
    df_val[col] = np.log1p(df_val[col])
    df_test[col] = np.log1p(df_test[col])


scaler = StandardScaler()
df_full_train[numerical] = scaler.fit_transform(df_full_train[numerical])
df_train[numerical] = scaler.fit_transform(df_train[numerical])
df_val[numerical] = scaler.transform(df_val[numerical])
df_test[numerical] = scaler.transform(df_test[numerical])


# XGBoost:


# Train XGBoost

print("Training XGBoost Model... \n\n")


def xgb_train(df_train, df_val, y_train_log, y_val_log, eta, 
              num_boost_round, max_depth, min_child_weight):

    dicts_train = df_train.to_dict(orient='records')
    dicts_val = df_val.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts_train)
    X_val = dv.transform(dicts_val)

    features = dv.get_feature_names_out().tolist()

    dtrain = xgb.DMatrix(X_train, label=y_train_log, feature_names=features)
    dval = xgb.DMatrix(X_val, label=y_val_log, feature_names=features)

    watchlist = [(dtrain, 'train'), (dval, 'val')]

    xgb_params = {
        'eta': eta,                     
        'max_depth': max_depth,                
        'min_child_weight': min_child_weight,         
        'objective': 'reg:squarederror',                        
        'eval_metric':['rmse', 'mae'],
        'nthreads':8,         
        'seed':1,            
        'verbosity':0  
    }


    evals_result = {}

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Predictions in log scale
    y_train_pred_log = model.predict(dtrain)
    y_val_pred_log = model.predict(dval)

    # Convert back to original scale
    y_train_pred = np.expm1(y_train_pred_log)
    y_val_pred = np.expm1(y_val_pred_log)

    # Compute metrics in original scale
    df_metrics = pd.DataFrame({
    'boost_round': range(num_boost_round),
    'train_rmse': evals_result['train']['rmse'],
    'val_rmse': evals_result['val']['rmse'],
    'train_mae': evals_result['train']['mae'],
    'val_mae': evals_result['val']['mae']
    })


    return dv, model, df_metrics



# Predict function
def xgb_predict(df, dv, model):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)

    features = dv.get_feature_names_out().tolist()

    d = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(d)

    return y_pred



def regression_metrics(y_actual, y_pred):
    mse = mean_squared_error(y_actual, y_pred)  # compare with y_val in original units
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    return rmse, mae, r2

print("Evaluation Metrtics on Validation Dataset... \n\n")

dv, model, df_metrics = xgb_train(
        df_train, df_val, y_train_log, y_val_log,
        eta=eta, num_boost_round=200, max_depth=max_depth, min_child_weight=min_child_weight
    )


y_pred_val_log = xgb_predict(df_val, dv, model)
y_pred_val = np.expm1(y_pred_val_log) 


rmse, mae, r2_val = regression_metrics(y_val, y_pred_val)

print(f"Validation RMSE:{rmse}")
print(f"Validation MAE:{mae}")
print(f"Validation R2:{r2_val}")



def xgb_train_full(df_train, y_train_log, eta, 
              num_boost_round, max_depth, min_child_weight):

    dicts_train = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts_train)

    features = dv.get_feature_names_out().tolist()

    dtrain = xgb.DMatrix(X_train, label=y_train_log, feature_names=features)

    xgb_params = {
        'eta': eta,                     
        'max_depth': max_depth,                
        'min_child_weight': min_child_weight,         
        'objective': 'reg:squarederror',                        
        'eval_metric': ['rmse', 'mae'],
        'nthreads': 8,         
        'seed': 1,            
        'verbosity': 0  
    }

    evals_result = {}
    watchlist = [(dtrain, 'train')]  

    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Gather metrics
    df_metrics = pd.DataFrame({
        'boost_round': range(num_boost_round),
        'train_rmse': evals_result['train']['rmse'],
        'train_mae': evals_result['train']['mae']
    })

    return dv, model, df_metrics

print("Training the full train dataframe... \n\n")

dv, model, df_metrics = xgb_train_full(
        df_full_train, y_full_train_log, eta=eta, num_boost_round=200,
        max_depth=max_depth, min_child_weight=min_child_weight
    )

# Predict on test set
y_pred_test_log = xgb_predict(df_test, dv, model)
y_pred_test = np.expm1(y_pred_test_log)  # convert back to original units

print("Evaluation Metrics on Test... \n\n")

# Evaluate metrics on test
rmse, mae, r2 = regression_metrics(y_test, y_pred_test)

print(f"Test RMSE:{rmse}")
print(f"Test MAE:{mae}")
print(f"Test R2:{r2}")

print("Saving the final model in local folder... \n\n")

output_file='../model/xgboost_eta=%s_depth=%s_minchild=%s_round=200.bin'%(eta, max_depth, min_child_weight)
with open(output_file, 'wb') as f_out:
    pickle.dump((model,dv), f_out)
    






