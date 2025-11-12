# Use official Python slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for some Python packages
RUN apt-get update && apt-get install -y build-essential gcc libffi-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Pipenv and Gunicorn
RUN pip install --no-cache-dir pipenv gunicorn

# Copy Pipfile and Pipfile.lock and install dependencies
COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy --ignore-pipfile

# Copy application code
COPY scripts/ ./scripts
COPY templates/ ./templates
COPY static/ ./static
COPY model/ ./model

# Expose port
EXPOSE 7860

# Run app with Gunicorn
# --chdir ensures Gunicorn starts in scripts so "serve:app" works
CMD ["gunicorn", "--chdir", "scripts", "--bind", "0.0.0.0:7860", "serve:app", "--workers", "3", "--threads", "2"]
