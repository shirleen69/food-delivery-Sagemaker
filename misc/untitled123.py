import pytest
import pandas as pd
import numpy as np
import joblib
from moto import mock_aws
import boto3
from botocore.exceptions import ClientError
import os
import sys

# Add the 'scripts' directory to the Python path
current_dir = os.path.dirname(__file__)
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
sys.path.insert(0, scripts_dir)

from delivery_inference import (preprocess_input, load_prediction_data, load_prediction_data_from_s3, 
                         load_model, load_model_from_s3, predict_delivery_time, 
                         save_prediction_results, save_prediction_results_to_s3)

# Constants
LOCAL_MODEL_FILE = 'temp_model.pkl'
LOCAL_PREDICTION_FILE = 'temp_prediction_data.csv'
LOCAL_PREDICTION_RESULTS_FILE = 'temp_prediction_results.csv'

# Helper functions
def create_dummy_model():
    from sklearn.dummy import DummyRegressor
    model = DummyRegressor(strategy="mean")
    model.fit([[0]], [0])  # Dummy fit
    joblib.dump(model, LOCAL_MODEL_FILE)
    return model

def create_dummy_data():
    data = pd.DataFrame({
        'Age_of_Delivery_Partner': [25, 30],
        'Ratings_of_Previous_Deliveries': [4.5, 4.0],
        'Total_Distance': [5.0, 3.0]
    })
    data.to_csv(LOCAL_PREDICTION_FILE, index=False)
    return data

@pytest.fixture
def dummy_model():
    return create_dummy_model()

@pytest.fixture
def dummy_data():
    return create_dummy_data()

def test_preprocess_input():
    features = [1, 2, 3]
    processed_features = preprocess_input(features)
    assert processed_features.shape == (1, 3, 1)

def test_load_prediction_data(dummy_data):
    data = load_prediction_data(LOCAL_PREDICTION_FILE)
    pd.testing.assert_frame_equal(data, dummy_data)

def test_load_prediction_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_prediction_data('non_existent_file.csv')

def test_load_prediction_data_from_s3(dummy_data):
    with mock_aws():
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'dummy_bucket'
        file_key = 'dummy_key'
        
        s3.create_bucket(Bucket=bucket_name)
        s3.put_object(Bucket=bucket_name, Key=file_key, Body=dummy_data.to_csv(index=False))

        data = load_prediction_data_from_s3(bucket_name, file_key)
        pd.testing.assert_frame_equal(data, dummy_data)

def test_load_prediction_data_from_s3_file_not_found():
    with mock_aws():
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'dummy_bucket'
        
        s3.create_bucket(Bucket=bucket_name)

        data = load_prediction_data_from_s3(bucket_name, 'non_existent_key')
        assert data is None

def test_load_model(dummy_model):
    loaded_model = load_model(LOCAL_MODEL_FILE)
    assert loaded_model is not None

def test_load_model_from_s3(dummy_model):
    with mock_aws():
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'dummy_bucket'
        file_key = 'dummy_key'

        # Create a bucket and upload the dummy model
        s3.create_bucket(Bucket=bucket_name)
        with open(LOCAL_MODEL_FILE, 'wb') as f:
            joblib.dump(dummy_model, f)
        s3.upload_file(LOCAL_MODEL_FILE, bucket_name, file_key)

        # Test loading the model from S3
        model = load_model_from_s3(bucket_name, file_key)
        assert model is not None

def test_predict_delivery_time(dummy_model, dummy_data):
    results = predict_delivery_time(dummy_model, dummy_data)
    assert 'Predicted_Delivery_Time' in results.columns

def test_save_prediction_results(dummy_data, tmpdir):
    result_csv = tmpdir.join("results.csv")
    result_json = tmpdir.join("results.json")

    save_prediction_results(dummy_data, result_csv, result_json)

    assert os.path.exists(result_csv)
    assert os.path.exists(result_json)

def test_save_prediction_results_to_s3(dummy_data):
    with mock_aws():
        s3 = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'dummy_bucket'
        s3.create_bucket(Bucket=bucket_name)

        save_prediction_results_to_s3(dummy_data, bucket_name, 'dummy_key_base')

        response_csv = s3.get_object(Bucket=bucket_name, Key='dummy_key_base.csv')
        response_json = s3.get_object(Bucket=bucket_name, Key='dummy_key_base.json')

        assert response_csv is not None
        assert response_json is not None


if __name__ == "__main__":
    # Directory where test results will be saved
    testresults_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
    if not os.path.exists(testresults_dir):
        os.makedirs(testresults_dir)

    # Run pytest with HTML reporting in the test-results directory
    report_path = os.path.join(testresults_dir, 'test_delivery_inference-report.html')
    pytest.main(['-v', '--html=' + report_path])