import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys
import pytest
import pandas as pd
from botocore.exceptions import ClientError

# Add the 'scripts' directory to the Python path
current_dir = os.path.dirname(__file__)
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
sys.path.insert(0, scripts_dir)

import delivery_inference

from delivery_inference import (
    load_model, load_prediction_data, load_model_from_s3,
    load_prediction_data_from_s3, predict_delivery_time,
    save_prediction_results, save_prediction_results_to_s3
)


# Define fixtures for paths
@pytest.fixture
def model_path():
    return os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_delivery_model.pkl')

@pytest.fixture
def data_path():
    return os.path.join(os.path.dirname(__file__), '..', 'data', 'delivery_prediction_data.csv')

@pytest.fixture
def results_dir():
    return os.path.join(os.path.dirname(__file__), '..', 'results')

@pytest.fixture
def result1_file_path(results_dir):
    return os.path.join(results_dir, 'delivery_prediction_results.csv')

@pytest.fixture
def result2_file_path(results_dir):
    return os.path.join(results_dir, 'delivery_prediction_results.json')

@pytest.fixture
def bucket_name():
    return 'interns-aws-experimentation-bucket'

@pytest.fixture
def model_file_key():
    return 'models/trained_delivery_model.pkl'

@pytest.fixture
def prediction_data_file_key():
    return 'models/delivery_prediction_data.csv'

@pytest.fixture
def prediction_results_file_key():
    return 'models/delivery_prediction_results'

# Test cases
def test_load_model(model_path):
    model = load_model(model_path)
    assert model is not None, "Failed to load the model"

def test_load_prediction_data(data_path):
    data = load_prediction_data(data_path)
    assert data is not None, "Failed to load prediction data"
    assert isinstance(data, pd.DataFrame), "Loaded data is not a DataFrame"

"""def test_load_model_from_s3(bucket_name, model_file_key):
    try:
        model = load_model_from_s3(bucket_name, model_file_key)
        assert model is not None, "Failed to load the model from S3"
    except ClientError as e:
        pytest.fail(f"ClientError: {e}")

def test_load_prediction_data_from_s3(bucket_name, prediction_data_file_key):
    try:
        data = load_prediction_data_from_s3(bucket_name, prediction_data_file_key)
        assert data is not None, "Failed to load prediction data from S3"
        assert isinstance(data, pd.DataFrame), "Loaded data from S3 is not a DataFrame"
    except ClientError as e:
        pytest.fail(f"ClientError: {e}")"""

def test_predict_delivery_time(model_path, data_path):
    model = load_model(model_path)
    data = load_prediction_data(data_path)
    assert model is not None, "Model is required for prediction"
    assert data is not None, "Data is required for prediction"
    
    prediction_results = predict_delivery_time(model, data)
    assert 'Predicted_Delivery_Time' in prediction_results.columns, "Predicted_Delivery_Time column not found"
    assert (prediction_results['Predicted_Delivery_Time'] > 0).all(), "Predicted delivery time should be greater than zero"

def test_save_prediction_results(results_dir, result1_file_path, result2_file_path):
    results = pd.DataFrame({
        'Age_of_Delivery_Partner': [25, 30],
        'Ratings_of_Previous_Deliveries': [4.5, 4.0],
        'Total_Distance': [12.3, 15.2],
        'Predicted_Delivery_Time': [10, 12]
    })
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    save_prediction_results(results, result1_file_path, result2_file_path)
    
    assert os.path.exists(result1_file_path), f"Failed to save CSV results to {result1_file_path}"
    assert os.path.exists(result2_file_path), f"Failed to save JSON results to {result2_file_path}"

    # Check CSV file content
    saved_csv_data = pd.read_csv(result1_file_path)
    assert saved_csv_data.shape == results.shape, f"CSV results shape mismatch. Expected {results.shape}, got {saved_csv_data.shape}"

    # Check JSON file content
    saved_json_data = pd.read_json(result2_file_path, orient='records')
    assert saved_json_data.shape == results.shape, f"JSON results shape mismatch. Expected {results.shape}, got {saved_json_data.shape}"

"""def test_save_prediction_results_to_s3(results_dir, bucket_name, prediction_results_file_key):
    results = pd.DataFrame({
        'Age_of_Delivery_Partner': [25, 30],
        'Ratings_of_Previous_Deliveries': [4.5, 4.0],
        'Total_Distance': [12.3, 15.2],
        'Predicted_Delivery_Time': [10, 12]
    })
    
    # Save results to S3
    try:
        save_prediction_results_to_s3(results, bucket_name, prediction_results_file_key)
    except ClientError as e:
        pytest.fail(f"ClientError: {e}")

    # You can add additional checks here if needed to verify the existence of the files in S3
"""


if __name__ == "__main__":
    # Directory where test results will be saved
    testresults_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
    if not os.path.exists(testresults_dir):
        os.makedirs(testresults_dir)

    # Run pytest with coverage reporting
    pytest_args = [
        '--cov=delivery_train', # Cover specific module
        '--cov-report=html',               # Generate HTML report
        '--cov-report=term-missing',       # Show summary in terminal
        '-v',                              # Verbose output
        f'--html={os.path.join(testresults_dir, "test_delivery_inference-report.html")}'
    ]
    pytest.main(pytest_args)


