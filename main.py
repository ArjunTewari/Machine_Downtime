# import pandas as pd
# import numpy as np
#
# df = pd.DataFrame(pd.read_csv("Machine Downtime.csv")).set_index("Date")
#
# df_encoded = pd.get_dummies(df, columns=['Machine_ID', 'Assembly_Line_No'])
#
# df_encoded = df_encoded.replace({'Downtime': {'Machine_Failure': 1.0, 'No_Machine_Failure': 0.0}})
#
# df_encoded.info()
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
#
# scaler_1 = MinMaxScaler()
# df_encoded[df_encoded.select_dtypes("number").columns] = scaler_1.fit_transform(df_encoded.select_dtypes("number"))
#
# df_encoded.dropna(inplace=True)
#
#
# x = df_encoded.drop(columns=['Downtime'])
# y = df_encoded["Downtime"]
#
# x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)
#
# from sklearn.linear_model import LogisticRegression
# model_1 = LogisticRegression()
# model_1.fit(x_train, y_train)
# probabilites = model_1.predict_proba(x_test)[:, 1]
#
# threshold = 0.7
# predictions = (probabilites >= threshold).astype(int)
# final_result = pd.DataFrame({"Confidence":probabilites.round(3), "Machine Failure":predictions})
# final_result.replace({'Machine Failure': {1: "Yes", 0: "No"}}, inplace=True)
#
# from sklearn.metrics import accuracy_score, precision_score, f1_score
# print(accuracy_score(y_test,predictions))
# print(precision_score(y_test,predictions))
# print(f1_score(y_test, predictions))

from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

app = Flask(__name__)
data_path = "uploaded_data.csv"
model_path = "model.pkl"

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Predictive API. Use /upload, /train, or /predict endpoints."})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        file.save(data_path)
        return jsonify({"message": "File uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        if not os.path.exists(data_path):
            return jsonify({"error": "No data uploaded yet"}), 400

        # Load data
        data = pd.read_csv(data_path)
        if not {'Coolant_Temperature', 'Spindle_Speed(RPM)', 'Downtime'}.issubset(data.columns):
            return jsonify({"error": "Dataset missing required columns"}), 400
        data.dropna(inplace=True)
        # Prepare data
        X = data[['Coolant_Temperature', 'Spindle_Speed(RPM)']]
        y = data['Downtime']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Save model
        joblib.dump(model, model_path)

        return jsonify({"message": "Model trained successfully", "accuracy": accuracy}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not trained yet"}), 400

        # Load model
        model = joblib.load(model_path)

        # Get input data
        data = request.get_json()
        if not data or not {'Coolant_Temperature', 'Spindle_Speed(RPM)'}.issubset(data):
            return jsonify({"error": "Invalid input format. Required keys: 'Coolant_Temperature', 'Spindle_Speed(RPM)'"}), 400

        # Make prediction
        input_data = [[data['Coolant_Temperature'], data['Spindle_Speed(RPM)']]]
        prediction = model.predict(input_data)[0]
        confidence = max(model.predict_proba(input_data)[0])

        return jsonify({"Downtime": "Yes" if prediction == 1 else "No", "Confidence": confidence}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)