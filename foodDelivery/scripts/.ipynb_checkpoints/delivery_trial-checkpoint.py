import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib
import boto3
import os
from botocore.exceptions import ClientError

# Constants
R = 6371  # Earth's radius in kilometers
LOCAL_MODEL_FILE = 'temp_model.pkl'  # Local file path for temporary storage

def load_data_from_s3(bucket_name, data_file_key):
    """Load the data from a CSV file stored in S3."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=data_file_key)
    data = pd.read_csv(obj['Body'])
    return data


def save_model_to_s3(model, bucket_name, model_file_key):
    """Save the trained model to a pickle file in S3."""
    joblib.dump(model, LOCAL_MODEL_FILE)
    s3 = boto3.client('s3')
    s3.upload_file(LOCAL_MODEL_FILE, bucket_name, model_file_key)
    os.remove(LOCAL_MODEL_FILE)  # Clean up the local file


def load_model_from_s3(bucket_name, model_file_key):
    """Load the trained model from a pickle file stored in S3."""
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=model_file_key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False


def explore_data(data):
    """Print data head, info, and check for missing values."""
    print(data.head())
    data.info()
    print(data.isnull().sum())


def deg_to_rad(degrees):
    """Convert degrees to radians."""
    return degrees * (np.pi / 180)


def distcalculate(lat1, lon1, lat2, lon2):
    """Calculate the distance between two points using the haversine formula."""
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lat2 - lon1)
    a = np.sin(d_lat / 2)**2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def calculate_distances(data):
    """Calculate the distance between each pair of points."""
    data['distance'] = np.nan
    for i in range(len(data)):
        data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'],
                                                data.loc[i, 'Restaurant_longitude'],
                                                data.loc[i, 'Delivery_location_latitude'],
                                                data.loc[i, 'Delivery_location_longitude'])
    return data


def preprocess_data(data):
    """Preprocess the data (e.g., handle missing values, calculate distances)."""
    data = calculate_distances(data)
    return data


def split_data(data):
    """Split the data into training and testing sets."""
    x = np.array(data[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]])
    y = np.array(data[["Time_taken(min)"]])
    return train_test_split(x, y, test_size=0.10, random_state=42)


def create_model(input_shape):
    """Create and compile the LSTM model."""
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(model, x_train, y_train):
    """Train the LSTM model."""
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model


def predict_delivery_time(model):
    """Predict the delivery time based on user input."""
    print("Food Delivery Time Prediction")
    a = int(input("Age of Delivery Partner: "))
    b = float(input("Ratings of Previous Deliveries: "))
    c = int(input("Total Distance: "))

    features = np.array([[a, b, c]])
    features = features.reshape((features.shape[0], features.shape[1], 1))  # Reshape for LSTM
    print("Predicted Delivery Time in Minutes = ", model.predict(features))


if __name__ == "__main__":
    # Define S3 bucket and file details
    bucket_name = 'interns-aws-experimentation-bucket'
    data_file_key = 'deliverytime.txt'
    model_file_key = 'models/trained_delivery_model.pkl'

    # Load data from S3
    data = load_data_from_s3(bucket_name, data_file_key)

    # Explore data
    explore_data(data)

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    x_train, x_test, y_train, y_test = split_data(data)

    # Reshape data for LSTM input
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    if load_model_from_s3(bucket_name, model_file_key):
        # Attempt to load the model from S3
        print("fILE ALREADY EXISTS")
    else:
        # If loading fails, create and train a new model
        print("Training a new model.")
        model = create_model((x_train.shape[1], 1))
        model = train_model(model, x_train, y_train)
        # Save the new model to S3
        save_model_to_s3(model, bucket_name, model_file_key)
        print("Model trained and saved to S3.")

    # Predict delivery time
    predict_delivery_time(model)
