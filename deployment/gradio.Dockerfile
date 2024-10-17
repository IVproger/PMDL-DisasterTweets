# Use the official Python 3.11 slim image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install all dependencies 
RUN pip3 install gradio requests

# Copy the gradio_app.py file into the container at /app
COPY gradio_app.py .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Command to run the gradio app
CMD ["bash", "-c", "python3 gradio_app.py"]