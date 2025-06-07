from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(_name_)
model = joblib.load('xgb_model.pkl')

FEATURES = [
    'Glucose',
    'Glucose_Insulin_Interaction',
    'BMI_Age_Risk',
    'BMI',
    'Age',
    'DiabetesPedigreeFunction',
    'BloodPressure',
    'Pregnancies',
    'SkinThickness',
    'Insulin'
]

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    try:
        X = np.array([data[feat] for feat in FEATURES], dtype=float).reshape(1, -1)
        pred = model.predict(X)[0]
        label = "High Risk of Diabetes" if pred == 1 else "Low Risk of Diabetes"
        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'prediction': f"Error: {str(e)}"})

if _name_ == '_main_':
    app.run(debug=True)gi