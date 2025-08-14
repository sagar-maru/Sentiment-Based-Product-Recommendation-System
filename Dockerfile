# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (better build cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default environment variables
ENV PYTHONUNBUFFERED=1

# Flask app environment
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Run Flask via Gunicorn in production mode
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]