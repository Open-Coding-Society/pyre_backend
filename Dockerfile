FROM docker.io/python:3.12

# Set working directory
WORKDIR /home/ubuntu/pyre_backend

# Update system packages (Python is already installed in base image)
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Run database scripts (with python interpreter)
RUN python ./scripts/db_backup.py
RUN echo "y" | python ./scripts/db_init.py
RUN python ./scripts/db_restore.py

# Set environment variables
ENV GUNICORN_CMD_ARGS="--workers=3 --bind=0.0.0.0:8505"
ENV FLASK_ENV=deployed

# Expose port
EXPOSE 8505

# Start application
CMD ["gunicorn", "main:app"]