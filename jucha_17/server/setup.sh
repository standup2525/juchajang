#!/bin/bash

echo "[SETUP] Starting server initialization..."

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install required packages
pip install --upgrade pip
pip install flask sense-hat

# 3. Check directory and file creation
mkdir -p templates
touch plate_log.csv
echo "Waiting for entry" > status.txt

echo "[SETUP] Done! Run 'source venv/bin/activate' and then 'python server.py'." 