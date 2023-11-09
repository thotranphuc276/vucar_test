import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib
import os
from train import predict_new
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Get the path to the model folder
model_folder = os.path.join(os.path.dirname(__file__), 'model')
df = pd.read_csv('car.csv')
df.drop(columns=['id', 'list_id', 'list_time', 'price'], inplace=True)
df.drop(df[df['manufacture_date'] < 0].index, inplace=True)
df.drop(df[df['seats'] < 0].index, inplace=True)

# Get unique values
def get_unique_values(feature):
    return np.sort(df[feature].dropna().unique())
unique_values = {feature: get_unique_values(feature) for feature in df.columns}
unique_values['seats'] = unique_values['seats'].astype(int)

# Load the RandomForestRegressor model
model_path = os.path.join(model_folder, 'random_forest_model.joblib')
model = joblib.load(model_path)
model2_path = os.path.join(model_folder, 'random_forest_model_ori.joblib')
model2 = joblib.load(model_path)

# Get encoder
new_car_data = pd.read_csv('new_car.csv')
encoder = {}
for column in new_car_data.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    encoder[column] = label_encoder.fit(new_car_data[column])


@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request form
        data = {
            'manufacture_date': int(request.form['manufacture_date']),
            'brand': request.form['brand'],
            'model': request.form['model'],
            'origin': request.form['origin'],
            'gearbox': request.form['gearbox'],
            'fuel': request.form['fuel'],
            'mileage_v2': int(request.form['mileage_v2']),
            'condition': request.form['condition'],
            'type': request.form['type'],
            'seats': int(request.form['seats'])
        }

        # Make a prediction
        if data['origin'] == '':
            data.pop('origin')
            prediction = predict_new(model, encoder, data)
        else:
            prediction = predict_new(model2, encoder, data)

        return render_template('prediction_results.html', prediction=prediction, features = data)
    except Exception as e:
        return render_template('index.html', error=f'Error: {str(e)}', unique_values=unique_values)

