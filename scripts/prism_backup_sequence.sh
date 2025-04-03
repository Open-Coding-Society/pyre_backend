#!/bin/bash

cd /home/ubuntu/QcommVNE_backend

# Verify the installation and check the Python version
python --version

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

cd /home/ubuntu/QcommVNE_backend
./scripts/db_backup.py