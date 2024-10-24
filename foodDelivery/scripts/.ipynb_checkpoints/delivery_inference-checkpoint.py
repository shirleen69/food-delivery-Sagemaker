import pandas as pd
import numpy as np
import joblib
import boto3
from botocore.exceptions import ClientError
import os

LOCAL_MODEL_FILE = 'temp_model.pkl'
LOCAL_PREDICTION_FILE = 'temp_prediction_data.csv'
LOCAL_PREDICTION_RESULTS_FILE = 'temp_prediction_results.csv'


def preprocess_input(features):
    """Preprocess input data."""
    features = np.array(features).reshape((1, -1, 1))  # Reshape for LSTM
    return features


def load_prediction_data(file_path):
    """Load the data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def load_prediction_data_from_s3(bucket_name, prediction_data_file_key):
    """Load the data from a CSV file stored in S3."""
    s3 = boto3.client('s3')
    try:
        # Download the data file from S3
        with open(LOCAL_PREDICTION_FILE, 'wb') as f:
            s3.download_fileobj(bucket_name, prediction_data_file_key, f)

        # Load the data from the downloaded file
        data = pd.read_csv(LOCAL_PREDICTION_FILE)
        print("Prediction data loaded successfully.")
        print(data.head())  # Debug: Print the head of the dataframe
        return data
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Prediction data file not found in S3.")
            return None
        else:
            raise


def load_model(filename):
    """Load the trained model from the pickle file."""
    return joblib.load(filename)


def load_model_from_s3(bucket_name, model_file_key):
    """Load the trained model from a joblib file stored in S3."""
    s3 = boto3.client('s3')
    try:
        # Download the model file from S3
        with open(LOCAL_MODEL_FILE, 'wb') as f:
            s3.download_fileobj(bucket_name, model_file_key, f)

        # Load the model from the downloaded file
        model = joblib.load(LOCAL_MODEL_FILE)
        print("Model loaded successfully.")
        return model
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            print("Model file not found in S3.")
            return None
        else:
            raise


def predict_delivery_time(model, data):
    """Predict the delivery time based on user input."""
    features = data[['Age_of_Delivery_Partner', 'Ratings_of_Previous_Deliveries', 'Total_Distance']].values
    features = features.reshape((features.shape[0], features.shape[1], 1))  # Reshape for LSTM
    predictions = model.predict(features)
    data['Predicted_Delivery_Time'] = predictions
    print("Predictions made successfully.")
    print(data.head())  # Debug: Print the head of the dataframe with predictions
    return data


def save_prediction_results(results, result1_file_path, result2_file_path):
    """Save the prediction results to both CSV and JSON files locally."""
    # Convert DataFrame to CSV
    results.to_csv(result1_file_path, index=False)
    print(f"CSV results saved to: {result1_file_path}")

    # Convert DataFrame to JSON
    results.to_json(result2_file_path, orient='records')
    print(f"JSON results saved to: {result2_file_path}")


def save_prediction_results_to_s3(results, bucket_name, prediction_results_file_key_base):
    """Save the prediction results to both CSV and JSON files in S3."""
    # Convert DataFrame to CSV
    csv_data = results.to_csv(index=False)

    # Convert DataFrame to JSON
    json_data = results.to_json(orient='records')

    # Upload the CSV data to S3
    s3 = boto3.client('s3')
    csv_key = f"{prediction_results_file_key_base}.csv"
    json_key = f"{prediction_results_file_key_base}.json"

    try:
        # Save CSV file
        s3.put_object(Bucket=bucket_name, Key=csv_key, Body=csv_data)
        print(f"CSV results saved to S3 as: {csv_key}")

        # Save JSON file
        s3.put_object(Bucket=bucket_name, Key=json_key, Body=json_data)
        print(f"JSON results saved to S3 as: {json_key}")

    except ClientError as e:
        print("Error uploading the file to S3:", e)
        raise


def main():
    """ # Define S3 bucket and file details
    bucket_name = 'interns-aws-experimentation-bucket'
    model_file_key = 'models/trained_delivery_model.pkl'
    prediction_data_file_key = 'models/delivery_prediction_data.csv'
    prediction_results_file_key = 'models/delivery_prediction_results.csv'

    # Load the trained model
    model = load_model_from_s3(bucket_name, model_file_key)
    if model is None:
        print(f"Model file {model_file_key} not found. Please ensure the model is trained and saved as {model_file_key}.")
        exit()

    # Load the prediction data
    data = load_prediction_data_from_s3(bucket_name, prediction_data_file_key)
    if data is None:
        print(f"Prediction data file {prediction_data_file_key} not found. Please ensure the data is available in {prediction_data_file_key}.")
        exit()

    # Make predictions
    prediction_results = predict_delivery_time(model, data)

    # Save prediction results to S3
    save_prediction_results_to_s3(prediction_results, bucket_name, prediction_results_file_key)

    print("Predictions saved to S3.")"""


# Define local file paths
model_filename = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_delivery_model.pkl')
data_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'delivery_prediction_data.csv')
results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
result1_file_path = os.path.join(results_dir, 'delivery_prediction_results.csv')
result2_file_path = os.path.join(results_dir, 'delivery_prediction_results.json')

# Check if results directory exists
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load model locally
model = load_model(model_filename)
if model is None:
    print(f"Model file {model_filename} not found. Please ensure the model is trained and saved as {model_filename}.")
    exit()

# Load prediction data locally
data = load_prediction_data(data_filename)
if data is None:
    print(f"Prediction data file {data_filename} not found. Please ensure the data is available as {data_filename}.")
    exit()

# Make predictions locally
prediction_results = predict_delivery_time(model, data)

# Save prediction results locally
save_prediction_results(prediction_results, result1_file_path, result2_file_path)

print("Prediction results saved successfully!!")


if __name__ == "__main__":
    main()
