import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import joblib
import boto3
import os

# Constants
R = 6371  # Earth's radius in kilometers


def load_data(file_path):
    """Load the data from a CSV file."""
    return pd.read_csv(file_path)


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


def save_model(model, filename):
    """Save the trained model as a pickle file."""
    joblib.dump(model, filename)


def load_model(filename):
    """Load the trained model from the pickle file."""
    return joblib.load(filename)


def preprocess_input(features):
    """Preprocess input data."""
    return np.array(features)


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
    # Load data
    data = load_data("/food-delivery/data/deliverytime.txt")

    # Explore data
    explore_data(data)

    # Preprocess data
    data = preprocess_data(data)

    # Visualize data
    visualize_data(data)

    # Split data
    x_train, x_test, y_train, y_test = split_data(data)

    # Reshape data for LSTM input
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Create and train model
    model = create_model((x_train.shape[1], 1))
    model = train_model(model, x_train, y_train)

    # Train model if not already trained
    model_filename = "/food-delivery/trained_delivery_model.pkl"
    try:
        model = load_model(model_filename)
    except FileNotFoundError:
        model = create_model((x_train.shape[1], 1))
        model = train_model(model, x_train, y_train)
        save_model(model, model_filename)

    # Predict delivery time
    predict_delivery_time(model)
