import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

def train(model, X_train, y_train, X_test, model_name, results_file):
    """
    Train the specified machine learning model, evaluate its performance on the test set,
    and record training and inference times in a results file.

    Parameters:
    - model (object): The machine learning model to train.
    - X_train (pd.DataFrame): Features of the training set.
    - y_train (pd.Series): Target variable of the training set.
    - X_test (pd.DataFrame): Features of the test set.
    - model_name (str): Name of the model.
    - results_file (str): File path to save training and inference results.

    Returns:
    - y_pred (np.ndarray): Predicted values on the test set.
    """

    start_time_train = time.time()
    model.fit(X_train, y_train)
    end_time_train = time.time()
    training_time = end_time_train - start_time_train

    print(f"\n\nTraining time: {training_time} seconds for {X_train.shape[0]} samples")

    start_time_inference = time.time()
    y_pred = model.predict(X_test)
    end_time_inference = time.time()
    inference_time = (end_time_inference - start_time_inference) / X_test.shape[0]

    print(f"Inference time for a sample: {inference_time} seconds")

    # Append results to a file
    with open(results_file, 'a') as f:
        f.write(f"\n\nModel: {model_name}\n")
        f.write(f"Training time: {training_time} seconds for {X_train.shape[0]} samples\n")
        f.write(f"Inference time for a sample: {inference_time} seconds\n")

    return y_pred

def evaluate(y_test, y_pred, model_name, results_file):
    """
    Evaluate the performance of a machine learning model using various metrics and
    record the results in a results file.

    Parameters:
    - y_test (pd.Series): True values of the target variable in the test set.
    - y_pred (np.ndarray): Predicted values on the test set.
    - model_name (str): Name of the model.
    - results_file (str): File path to save evaluation results.

    Returns:
    None
    """

    with open(results_file, 'a') as f:
        f.write(f"\t\tError Table {model_name}\n")
        f.write('Mean Absolute Error      : {}\n'.format(mean_absolute_error(y_test, y_pred)))
        f.write('Mean Squared  Error      : {}\n'.format(mean_squared_error(y_test, y_pred)))
        f.write('Root Mean Squared  Error : {}\n'.format(np.sqrt(mean_squared_error(y_test, y_pred))))
        f.write('R Squared Error          : {}\n'.format(r2_score(y_test, y_pred)))

def predict_new(model, encoder, X):
    """
    Make predictions using a trained machine learning model on new data.

    Parameters:
    - model (object): The trained machine learning model.
    - encoder (dict): Dictionary containing label encoders for categorical features.
    - X (pd.DataFrame): New data for prediction.

    Returns:
    - y_pred (np.ndarray): Predicted values on the new data.
    """

    X = pd.DataFrame(X, index=[0])
    for column in X.select_dtypes(include=['object']).columns:
        try:
            X[column] = encoder[column].transform(X[column])
        except:
            X[column] = -1
    return model.predict(X)

def main(path_file, results_file):
    """
    Main function to read data, train multiple machine learning models, and record results.

    Parameters:
    - path_file (str): File path to the input data.
    - results_file (str): File path to save training and evaluation results.

    Returns:
    None
    """
        
    new_car_data = pd.read_csv(path_file)

    X = new_car_data.drop([new_car_data.columns[0], 'price'], axis=1)
    y = new_car_data['price']

    # Handle categorical variables
    label_encoder = LabelEncoder()
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = label_encoder.fit_transform(X[column])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save result to file
    with open(results_file, 'a') as f:
        f.write(f"\n\nDataset: {path_file}\n")

    # Model 1
    linear_reg = LinearRegression()
    y_pred = train(linear_reg, X_train, y_train, X_test, 'Linear Regression', results_file)
    evaluate(y_test, y_pred, 'Linear Regression', results_file)

    # Model 2
    rf_model = RandomForestRegressor(n_estimators=20, random_state=42)
    y_pred = train(rf_model, X_train, y_train, X_test, 'Random Forest', results_file)
    evaluate(y_test, y_pred, 'Random Forest', results_file)

    # Model 3
    xgb_r = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
    y_pred = train(xgb_r, X_train, y_train, X_test, 'XGBoost', results_file)
    evaluate(y_test, y_pred, 'XGBoost', results_file)

    # Save model
    if path_file == 'new_car.csv':
        joblib.dump(rf_model, 'app/model/random_forest_model.joblib')
    if path_file == 'new_car_3.csv':
        joblib.dump(rf_model, 'app/model/random_forest_model_ori.joblib')

if __name__ == '__main__':
    results_file = 'results.txt'
    with open(results_file, 'w') as f:
        f.write("Model Training and Evaluation Results\n")

    main('new_car.csv', results_file)
    main('new_car_2.csv', results_file)
    main('new_car_3.csv', results_file)
