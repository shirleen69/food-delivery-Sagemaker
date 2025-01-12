import pytest
import pandas as pd
import numpy as np
import joblib
import os
from moto import mock_s3
from unittest.mock import patch
from tempfile import NamedTemporaryFile
from io import BytesIO
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM

import sys

from delivery_train import (
    load_data,
    load_data_from_s3,
    explore_data,
    deg_to_rad,
    distcalculate,
    calculate_distances,
    preprocess_data,
    split_data,
    create_model,
    train_model,
    save_model_to_s3,
    save_model,
)


# Add the 'scripts' directory to the Python path
current_dir = os.path.dirname(__file__)
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
sys.path.insert(0, scripts_dir)

BUCKET_NAME = 'interns-aws-experimentation-bucket'
DATA_KEY = 'deliverytime.txt'
MODEL_KEY = 'models/trained_delivery_model.pkl'


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    data = pd.DataFrame({
        'Restaurant_latitude': [12.9715987, 12.2958104],
        'Restaurant_longitude': [77.5945627, 76.6393805],
        'Delivery_location_latitude': [12.971619, 12.295811],
        'Delivery_location_longitude': [77.594563, 76.639381],
        'Time_taken(min)': [30, 40],
        'Delivery_person_Age': [25, 30],
        'Delivery_person_Ratings': [4.5, 4.7]
    })
    return data


@pytest.fixture
def s3_bucket(sample_data):
    """Fixture for a mock S3 bucket with sample data."""
    with mock_s3():
        s3 = boto3.client('s3', region_name='us-east-1')
        s3.create_bucket(Bucket=BUCKET_NAME)
        
        # Add a mock data file to the bucket
        data_csv = sample_data.to_csv(index=False)
        s3.put_object(Bucket=BUCKET_NAME, Key=DATA_KEY, Body=data_csv)
        
        yield s3


def test_load_data(sample_data):
    """Test for load_data function."""
    with NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
        sample_data.to_csv(temp_file.name, index=False)
        loaded_data = load_data(temp_file.name)
    assert loaded_data.equals(sample_data)


def test_load_data_from_s3(sample_data, s3_bucket):
    """Test for load_data_from_s3 function."""
    loaded_data = load_data_from_s3(BUCKET_NAME, DATA_KEY)
    assert loaded_data.equals(sample_data)


def test_explore_data(sample_data, capsys):
    """Test for explore_data function."""
    explore_data(sample_data)
    captured = capsys.readouterr()
    assert 'Restaurant_latitude' in captured.out
    assert 'Delivery_person_Age' in captured.out


def test_deg_to_rad():
    """Test for deg_to_rad function."""
    degrees = 180
    radians = deg_to_rad(degrees)
    assert radians == np.pi


def test_distcalculate():
    """Test for distcalculate function."""
    lat1, lon1 = 12.9715987, 77.5945627
    lat2, lon2 = 13.035, 77.597
    distance = distcalculate(lat1, lon1, lat2, lon2)
    assert distance > 0


def test_calculate_distances(sample_data):
    """Test for calculate_distances function."""
    calculated_data = calculate_distances(sample_data)
    assert 'distance' in calculated_data.columns
    assert not calculated_data['distance'].isnull().all()


def test_preprocess_data(sample_data):
    """Test for preprocess_data function."""
    preprocessed_data = preprocess_data(sample_data)
    assert 'distance' in preprocessed_data.columns


def test_split_data(sample_data):
    """Test for split_data function."""
    x_train, x_test, y_train, y_test = split_data(sample_data)
    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_create_model():
    """Test for create_model function."""
    input_shape = (3, 1)
    model = create_model(input_shape)
    assert isinstance(model, Sequential)
    assert len(model.layers) > 0


def test_train_model():
    """Test for train_model function."""
    input_shape = (3, 1)
    model = create_model(input_shape)
    x_train = np.random.rand(10, 3, 1)
    y_train = np.random.rand(10, 1)
    trained_model = train_model(model, x_train, y_train)
    assert trained_model is not None


@pytest.fixture
def mock_model():
    """Fixture for mock trained model."""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(3, 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def test_save_model_to_s3(mock_model, s3_bucket):
    """Test for save_model_to_s3 function."""
    save_model_to_s3(mock_model, BUCKET_NAME, MODEL_KEY)
    s3 = boto3.client('s3', region_name='us-east-1')
    response = s3.get_object(Bucket=BUCKET_NAME, Key=MODEL_KEY)
    assert response is not None
    assert response['Body'].read() is not None


def test_save_model(mock_model):
    """Test for save_model function."""
    with NamedTemporaryFile(delete=False) as temp_file:
        save_model(mock_model, temp_file.name)
        assert os.path.exists(temp_file.name)


if __name__ == "__main__":
    # Directory where test results will be saved
    testresults_dir = os.path.join(os.path.dirname(__file__), '..', 'test-results')
    if not os.path.exists(testresults_dir):
        os.makedirs(testresults_dir)

    # Run pytest with HTML reporting in the test-results directory
    report_path = os.path.join(testresults_dir, 'test_delivery_train-report.html')
    pytest.main(['-v', '--html=' + report_path])
