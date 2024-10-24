import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib
import boto3
import os
from botocore.exceptions import ClientError


R = 6371  # Earth's radius in kilometers
LOCAL_MODEL_FILE = 'temp_model.pkl'  # Local file path for temporary storage


def load_data(file_path):
    """Load the data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)


def load_data_from_s3(bucket_name, data_file_key):
    """Load the data from a CSV file stored in S3."""
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=data_file_key)
    data = pd.read_csv(obj['Body'])
    return data


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
    d_lon = deg_to_rad(lon2 - lon1)
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


def visualize_data(data):
    """Visualize the relationships between different features and the target variable."""
    scatter_plots = [
        ("distance", "Time_taken(min)", "Relationship Between Distance and Time Taken"),
        ("Delivery_person_Age", "Time_taken(min)", "Relationship Between Time Taken and Age", "distance"),
        ("Delivery_person_Ratings", "Time_taken(min)", "Relationship Between Time Taken and Ratings", "distance")
    ]
    for plot in scatter_plots:
        figure = px.scatter(data_frame=data, x=plot[0], y=plot[1], size="Time_taken(min)", trendline="ols", title=plot[2])
        if len(plot) == 4:
            figure.update_traces(marker=dict(color=data[plot[3]], colorscale='Viridis'))
        figure.show()

    fig = px.box(data, x="Type_of_vehicle", y="Time_taken(min)", color="Type_of_order")
    fig.show()


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
    model.fit(x_train, y_train, batch_size=1, epochs=9)
    return model


def save_model_to_s3(model, bucket_name, model_file_key):
    """Save the trained model to a pickle file in S3."""
    joblib.dump(model, LOCAL_MODEL_FILE)
    s3 = boto3.client('s3')
    s3.upload_file(LOCAL_MODEL_FILE, bucket_name, model_file_key)
    os.remove(LOCAL_MODEL_FILE)  # Clean up the local file


def save_model(model, filename):
    """Save the trained model as a pickle file."""
    joblib.dump(model, filename)


def main():

    # Define S3 bucket and file details
    bucket_name = 'interns-aws-experimentation-bucket'
    data_file_key = 'deliverytime.txt'
    model_file_key = 'models/trained_delivery_model.pkl'

    # Load data from S3
    #data = load_data_from_s3(bucket_name, data_file_key)

    # Define file paths
    data_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'deliverytime.txt')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_file_path = os.path.join(model_dir, 'trained_delivery_model.pkl')

    # Check if directories exist
    data_dir = os.path.dirname(data_file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Load data
    data = load_data(data_file_path)

    # Explore data
    explore_data(data)

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    x_train, x_test, y_train, y_test = split_data(data)

    # Reshape data for LSTM input
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # create and train a new model
    # Train model
    print("Training a new model.")
    model = create_model((x_train.shape[1], 1))
    model = train_model(model, x_train, y_train)
    save_model(model, model_file_path)
    print(f"Model trained and saved to {model_file_path}.")
    # Save the new model to S3
    #save_model_to_s3(model, bucket_name, model_file_key)
    #print("Model trained and saved to S3.")



if __name__ == "__main__":
    main()
