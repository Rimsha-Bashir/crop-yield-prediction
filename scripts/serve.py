import pickle
from flask import Flask 
from flask import request
from flask import jsonify 


eta = 0.1
max_depth = 7
min_child_weight = 5 

model_file = '../model/xgboost_eta=%s_depth=%s_minchild=%s_round=200.bin'%(eta, max_depth, min_child_weight)

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('Yield_Prediction')

@app.route('/predict', methods=['POST'])
def predict():
    '''write here'''


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
