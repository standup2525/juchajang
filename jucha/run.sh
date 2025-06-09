#!/bin/bash

# Set project folder path
PROJECT_ROOT="/home/pi/Desktop/jucha"

# Start virtual environment
source $PROJECT_ROOT/venv/bin/activate

# Set Python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT

# Create log folder
mkdir -p $PROJECT_ROOT/logs

# Check if AIHAT is connected
if [ -e /dev/mysidia0 ]; then
    echo "AIHAT detected"
else
    echo "Warning: AIHAT not detected, using CPU"
fi

# Check if camera is connected
if [ -e /dev/video0 ]; then
    echo "Camera detected"
else
    echo "Error: Camera not detected"
    exit 1
fi

# Run the system
cd $PROJECT_ROOT
python src/car_detection.py 