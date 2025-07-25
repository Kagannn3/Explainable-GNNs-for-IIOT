# Base image with Python 3.10 and basic OS utilities
FROM python:3.10-slim

# Set environment variables to prevent Python from buffering outputs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install essential OS-level dependencies
# - gcc: for building Python packages with C extensions
# - libpq-dev: for PostgreSQL support (optional)
# - git: in case you use pip install from GitHub
# - curl: for diagnostics (optional)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement definitions (used by docker-compose and manual build)
COPY docker/requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all project files into /app directory inside container
COPY . .

# Expose port if running a Flask/Django API (optional, else ignored)
EXPOSE 8000

# Default command can be overridden in docker-compose or manually
# Example below just starts a bash shell
CMD ["/bin/bash"]
