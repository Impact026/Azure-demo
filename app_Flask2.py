# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:24:14 2024

@author: Owner
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify

# Load the model from the file using read-binary ('rb') mode
with open("iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to my flask app!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input JSON from request
    features = np.array(data["features"]).reshape(1, -1)  # Reshape features
    predictions = model.predict(features)  # Predict using the loaded model
    return jsonify({"prediction": int(predictions[0])})  # Return prediction as JSON

if __name__ == "__main__":
    app.run(debug=True)