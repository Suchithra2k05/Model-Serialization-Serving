from flask import Flask, request, jsonify
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

# ------------------------------
# Step 1: Train and Save Model
# ------------------------------
model_filename = 'iris_model.pkl'

if not os.path.exists(model_filename):
    print("🔧 Training model and saving to disk...")

    # Load the Iris dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model using joblib
    joblib.dump(model, model_filename)

    print("✅ Model saved as 'iris_model.pkl'")
else:
    print("📦 Model already exists, loading from disk...")

# ------------------------------
# Step 2: Load Model and Serve via Flask
# ------------------------------
# Load the model
model = joblib.load(model_filename)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "🌸 Iris Classifier API is running! Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        features = data['features']  # Expecting a list of 4 numbers

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ------------------------------
# Step 3: Run the Flask app
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

# ------------------------------
# Step 1: Train and Save Model
# ------------------------------
model_filename = 'iris_model.pkl'

if not os.path.exists(model_filename):
    print("🔧 Training model and saving to disk...")

    # Load the Iris dataset
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Train Random Forest model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save model using joblib
    joblib.dump(model, model_filename)

    print("✅ Model saved as 'iris_model.pkl'")
else:
    print("📦 Model already exists, loading from disk...")

# ------------------------------
# Step 2: Load Model and Serve via Flask
# ------------------------------
# Load the model
model = joblib.load(model_filename)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "🌸 Iris Classifier API is running! Use POST /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.json
        features = data['features']  # Expecting a list of 4 numbers

        # Convert to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)
        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ------------------------------
# Step 3: Run the Flask app
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
