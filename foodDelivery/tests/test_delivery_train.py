import pytest
import pandas as pd
import numpy as np
import joblib
import os
import boto3
from tempfile import NamedTemporaryFile
from botocore.exceptions import ClientError
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import sys

# Add the 'scripts' directory to the Python path
current_dir = os.path.dirname(__file__)
scripts_dir = os.path.abspath(os.path.join(current_dir, '..', 'scripts'))
sys.path.insert(0, scripts_dir)


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

# Function to check the dataset
def check_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Check the number of rows and columns
    num_rows, num_columns = data.shape
    print(f"Number of rows: {num_rows}")
    print(f"Number of columns: {num_columns}")
    
    # Get the column names
    column_names = data.columns.tolist()
    print("Column names:")
    for column in column_names:
        print(column)
    return data

# Example usage of the check_dataset function
file_path = os.path.join(current_dir, '..', 'data', 'deliverytime.txt')
check_dataset(file_path)

# Expected data shape and columns
EXPECTED_NUM_ROWS = 45593
EXPECTED_NUM_COLUMNS = 11

@pytest.fixture
def expected_shape():
    return (EXPECTED_NUM_ROWS, EXPECTED_NUM_COLUMNS)

@pytest.fixture
def expected_columns_list():
    return [
        'ID',
        'Delivery_person_ID',
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'Restaurant_latitude',
        'Restaurant_longitude',
        'Delivery_location_latitude',
        'Delivery_location_longitude',
        'Type_of_order',
        'Type_of_vehicle',
        'Time_taken(min)'
    ]

def test_load_data(expected_shape, expected_columns_list):
    """Test for load_data function."""
    data_path = os.path.join(current_dir, '..', 'data', 'deliverytime.txt')
     # Check if the path exists
    assert os.path.exists(data_path), f"Data path '{data_path}' does not exist."
    loaded_data = load_data(data_path)
    assert loaded_data.shape == expected_shape, f"Expected shape {expected_shape}, but got {loaded_data.shape}"
    assert loaded_data.columns.tolist() == expected_columns_list, f"Expected columns {expected_columns_list}, but got {loaded_data.columns.tolist()}"
    assert not loaded_data.isnull().values.any(), "Data contains missing values"

"""
def test_load_data_from_s3(expected_shape, expected_columns_list):
    'Test for load_data_from_s3 function.'
    bucket_name = 'interns-aws-experimentation-bucket'
    key = 'deliverytime.txt'

    # Initialize S3 client
    s3 = boto3.client('s3')

    # Check if the bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            pytest.skip(f"Bucket '{bucket_name}' not found. Skipping test.")
        else:
            raise

    # Check if the object (key) exists in the bucket
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            pytest.skip(f"Key '{key}' not found in bucket '{bucket_name}'. Skipping test.")
        else:
            raise

    # Load data from S3
    response = s3.get_object(Bucket=bucket_name, Key=key)
    loaded_data = pd.read_csv(response['Body'])

    # Assertions
    assert loaded_data.shape == expected_shape, f"Expected shape {expected_shape}, but got {loaded_data.shape}"

    # Check if expected columns are a subset of the loaded columns
    loaded_columns = loaded_data.columns.tolist()
    missing_columns = [col for col in expected_columns_list if col not in loaded_columns]
    assert not missing_columns, f"Missing expected columns: {missing_columns}"

    assert not loaded_data.isnull().values.any(), "Data contains missing values"
"""

def test_deg_to_rad():
    """Test for deg_to_rad function."""
    degrees = 180
    radians = deg_to_rad(degrees)
    assert radians == np.pi
    
def test_deg_to_rad_zero_degrees():
    """Test deg_to_rad function with 0 degrees."""
    degrees = 0
    radians = deg_to_rad(degrees)
    assert radians == 0, "Expected 0 radians for 0 degrees"

def test_deg_to_rad_negative_degrees():
    """Test deg_to_rad function with negative degrees."""
    degrees = -90
    radians = deg_to_rad(degrees)
    assert radians == -np.pi / 2, "Expected -pi/2 radians for -90 degrees"

def test_deg_to_rad_positive_degrees():
    """Test deg_to_rad function with positive degrees."""
    degrees = 45
    radians = deg_to_rad(degrees)
    assert radians == np.pi / 4, "Expected pi/4 radians for 45 degrees"

def test_deg_to_rad_large_degrees():
    """Test deg_to_rad function with large degrees."""
    degrees = 720
    radians = deg_to_rad(degrees)
    assert radians == 4 * np.pi, "Expected 4*pi radians for 720 degrees"

def test_deg_to_rad_edge_cases():
    """Test deg_to_rad function with edge cases."""
    degrees = 360
    radians = deg_to_rad(degrees)
    assert radians == 2 * np.pi, "Expected 2*pi radians for 360 degrees"
    
    degrees = -360
    radians = deg_to_rad(degrees)
    assert radians == -2 * np.pi, "Expected -2*pi radians for -360 degrees"

def test_deg_to_rad_non_integer_degrees():
    """Test deg_to_rad function with non-integer degrees."""
    degrees = 30.5
    radians = deg_to_rad(degrees)
    assert radians == degrees * (np.pi / 180), f"Expected {degrees * (np.pi / 180)} radians for {degrees} degrees"


def test_distcalculate():
    """Test for distcalculate function."""
    lat1, lon1 = 12.9715987, 77.5945627
    lat2, lon2 = 13.035, 77.597
    distance = distcalculate(lat1, lon1, lat2, lon2)
    
    # Check if latitudes and longitudes are floats
    assert isinstance(lat1, float), f"Expected float for lat1, but got {type(lat1)}"
    assert isinstance(lon1, float), f"Expected float for lon1, but got {type(lon1)}"
    assert isinstance(lat2, float), f"Expected float for lat2, but got {type(lat2)}"
    assert isinstance(lon2, float), f"Expected float for lon2, but got {type(lon2)}"
    
    # Check if distance is a float
    assert isinstance(distance, float), f"Expected float for distance, but got {type(distance)}"
    
    assert distance > 0


def test_calculate_distances(expected_columns_list):
    """Test for calculate_distances function."""
    data_path = os.path.join(current_dir, '..', 'data', 'deliverytime.txt')
    loaded_data = load_data(data_path)
    calculated_data = calculate_distances(loaded_data)
    assert 'distance' in calculated_data.columns
    assert not calculated_data['distance'].isnull().all()

def test_preprocess_data(expected_columns_list):
    """Test for preprocess_data function."""
    data_path = os.path.join(current_dir, '..', 'data', 'deliverytime.txt')
    loaded_data = load_data(data_path)
    preprocessed_data = preprocess_data(loaded_data)
    assert 'distance' in preprocessed_data.columns

def test_split_data(expected_columns_list):
    """Test for split_data function."""
    data_path = os.path.join(current_dir, '..', 'data', 'deliverytime.txt')
    loaded_data = load_data(data_path)
    loaded_data['distance'] = 1.0  # Ensure 'distance' column exists
    x_train, x_test, y_train, y_test = split_data(loaded_data)
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
def trained_model():
    """Fixture for a trained model."""
    data_path = os.path.join(current_dir, '..', 'data', 'deliverytime.txt')
    loaded_data = load_data(data_path)
    preprocessed_data = preprocess_data(loaded_data)
    x_train, x_test, y_train, y_test = split_data(preprocessed_data)
    model = create_model((x_train.shape[1], 1))
    trained_model = train_model(model, x_train, y_train)
    return trained_model

"""def test_save_model_to_s3():
    'Test to check if the model exists in the S3 bucket and is a .pkl file.'
    bucket_name = 'interns-aws-experimentation-bucket'
    model_file_key = 'models/trained_delivery_model.pkl'

    # Verify the model exists in S3
    s3 = boto3.client('s3', region_name='us-east-1')

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_file_key)
    assert 'Contents' in response, f"Object with key '{model_file_key}' not found in bucket '{bucket_name}'"
    assert model_file_key.endswith('.pkl'), f"Expected model file key to end with '.pkl', but got '{model_file_key}'"
    print(f"Model '{model_file_key}' exists in S3 bucket '{bucket_name}' and is a .pkl file.")"""

def test_save_model():
    """Test to check if the model exists locally and is a .pkl file."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_file_path = os.path.join(model_dir, 'trained_delivery_model.pkl')

    # Verify the model exists locally
    assert os.path.exists(model_file_path), f"Expected model file at {model_file_path}, but file does not exist"
    assert model_file_path.endswith('.pkl'), f"Expected model file to end with '.pkl', but got '{model_file_path}'"
    print(f"Model saved successfully at {model_file_path} and is a .pkl file.")

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
        f'--html={os.path.join(testresults_dir, "test_delivery_train-report.html")}'
    ]
    pytest.main(pytest_args)
