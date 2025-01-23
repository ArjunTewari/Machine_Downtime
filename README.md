# Predictive API for Manufacturing Operations

This API predicts machine downtime using a manufacturing dataset. It includes endpoints to upload data, train a model, and make predictions.

## Features
- **Upload**: Upload a CSV file for training.
- **Train**: Train a machine learning model using the uploaded dataset.
- **Predict**: Predict machine downtime using input data.

## Requirements
- Python 3.7+
- Packages: `flask`, `pandas`, `scikit-learn`, `joblib`

Install dependencies:
```bash
pip install flask pandas scikit-learn joblib

Save the API code in app.py.

**Upload Dataset**: curl -X POST http://127.0.0.1:5000/upload -F "file=@path/to/your/dataset.csv"

**Train Model** : curl -X POST http://127.0.0.1:5000/train

**Predict Downtime** : curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"Coolant_Temperature": 80, "Spindle_Speed(RPM)": 120, "Torque(Nm)": 50, "Cutting(kN)": 10}'
