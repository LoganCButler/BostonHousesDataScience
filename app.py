
###   TODO   ###
import numpy as np
from flask import Flask, request, jsonify
import pickle
from pandas import DataFrame
import pandas as pd
import math

app = Flask(__name__)
# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route('/predict',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    
    asDf = pd.read_json(data['payload'])
    
    ## add Hyper columns to test data
    asDfHyper = asDf.copy()
    
    asDfHyper['hyper_A'] = (asDf[12]-asDf[5])
    asDfHyper['hyper_B'] = asDf[0].apply(abs).apply(math.sqrt)
    asDfHyper['hyper_C'] = asDf[12] < 0
            
    prediction = model.predict(asDfHyper)
    
    # Take the first value of prediction
    output = prediction[0][0]
    return jsonify({'prediction (10k): ': output })

if __name__ == '__main__':
    app.run(port=8081, debug=True)

################
