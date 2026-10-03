
# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application
COPY app.py .
COPY superkart_xgb_model.joblib .

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Run the application using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
