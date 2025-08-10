# Model-Serialization-Serving
📦 Model Serialization & Serving – Iris Classifier API
This project demonstrates the end-to-end process of training, saving, and serving a machine learning model using Python.
It combines model serialization with Flask API development to make predictions in real-time.

📌 Why Model Serialization Matters
In real-world machine learning:
Training a model repeatedly is costly in terms of computation and time.
Serialization allows saving the trained model to disk so it can be loaded instantly later.
The saved model can be deployed to production systems for real-time predictions.

Common Python serialization formats:
.pkl – Pickle format (can save any Python object, but less optimized for large NumPy arrays).
.joblib – More efficient for saving large numerical data (recommended for scikit-learn models).

📊 Project Overview
Dataset: Iris Dataset (from sklearn.datasets)
Algorithm: Random Forest Classifier
Serialization Library: joblib
API Framework: Flask
Purpose: Build a reusable and deployable prediction API

⚙️ Workflow

Step 1: Train & Save Model
Load the Iris dataset.
Train a Random Forest Classifier.
Save it to disk as iris_model.pkl using joblib.
Avoid retraining if the saved file already exists.

Step 2: Load Model
Load the serialized model into memory for predictions.

Step 3: Serve via Flask
Create a REST API with endpoints:
/ → Simple API status message.
/predict → Accepts JSON with 4 feature values, returns a class prediction.

Step 4: Make Predictions
Accepts JSON POST requests.
Converts input to NumPy arrays.
Returns predictions as JSON.

🔍 Example API Call
POST Request

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [6.1, 2.8, 4.7, 1.2]}'

JSON Response
json

{
    "prediction": 1
}
Class IDs in Iris dataset:
0 = Setosa, 1 = Versicolor, 2 = Virginica


📈 Advantages of This Approach
Fast Predictions – No need to retrain each time.
Lightweight Deployment – Minimal dependencies.
Reusable – One model can be served to multiple clients.
Extendable – Can integrate into cloud platforms like AWS, Azure, or Heroku.

💡 Key Learnings
How to serialize ML models with joblib.
How to serve ML models using Flask.
How to handle JSON requests and responses.
Importance of error handling in production APIs.

🛠 How to Run

# Clone the repository
git clone https://github.com/yourusername/model-serialization-serving.git
cd model-serialization-serving

# Install dependencies
pip install flask pandas scikit-learn joblib numpy

# Run the Flask app
python iris_flask_api.py
Visit: http://127.0.0.1:5000

🔮 Possible Improvements

Add model versioning to track changes.
Implement input validation for safer predictions.
Deploy the API on Heroku, Render, or AWS Lambda.
Add Swagger/OpenAPI documentation for better API usability.

