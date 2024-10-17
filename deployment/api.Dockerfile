# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY deployment/api_requirements.txt .

# Install the dependencies
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r api_requirements.txt

# Copy the source code into the container
COPY src ./src

# Copy the models into the container
COPY models ./models

# Copy the configs into the container
COPY configs ./configs

# Copy the app.py into the container
COPY deployment/app.py .

# Copy the healthcheck script into the container
COPY deployment/healthcheck.py ./deployment/healthcheck.py

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/app/src

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application with a debug step
CMD ["bash", "-c", "python3 app.py"]