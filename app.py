from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the model once at startup
try:
    model = joblib.load('xgb_model.pkl')
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define expected features
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
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json(force=True)

    try:
        # Check if all required features are present
        missing = [feat for feat in FEATURES if feat not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Prepare input for prediction
        X = np.array([data[feat] for feat in FEATURES], dtype=float).reshape(1, -1)

        # Perform prediction
        pred = model.predict(X)[0]
        label = "High Risk of Diabetes" if pred == 1 else "Low Risk of Diabetes"

        return jsonify({'prediction': label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app using Waitress for production
if __name__ == '__main__':
    try:
        from waitress import serve
        print("Starting production server with Waitress...")
        serve(app, host='0.0.0.0', port=5000)
    except ImportError:
        print("Waitress not installed, using Flask development server.")
        app.run(debug=True, use_reloader=False)
