# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 21:22:44 2024

@author: Owner
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load toy dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a simple model (Random Forest)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and check accuracy
y_pred = model.predict(X_test)
print("Model accuracy:", accuracy_score(y_test, y_pred))

import pickle
# Save the model using write-binary ('wb') mode
with open("iris_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)  # Pass the file object, not the filename



# import pickle
# import numpy as np
# from flask import Flask, request, jsonify

# # Load the model from the file using read-binary ('rb') mode
# with open("iris_model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return "Welcome to my flask app!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()  # Get input JSON from request
#     features = np.array(data["features"]).reshape(1, -1)  # Reshape features
#     predictions = model.predict(features)  # Predict using the loaded model
#     return jsonify({"prediction": int(predictions[0])})  # Return prediction as JSON

# if __name__ == "__main__":
#     app.run(debug=True)
