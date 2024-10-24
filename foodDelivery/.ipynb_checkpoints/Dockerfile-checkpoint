# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /opt/ml/processing/

# Copy the entire content of the foodDelivery directory into the container
COPY foodDelivery /opt/ml/processing/foodDelivery

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /opt/ml/processing/foodDelivery/requirements.txt

# Command to run when the container starts
ENTRYPOINT ["python", "/opt/ml/processing/foodDelivery/scripts/delivery_inference.py"]
