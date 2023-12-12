#!/bin/bash

# Name of the virtual environment
ENV_NAME="env-klaviyo-asgn-noah"

# Path to requirements.txt
REQUIREMENTS_PATH="./requirements.txt"

# Python script and its arguments
PYTHON_SCRIPT="./train_and_predict.py"
PYTHON_ARGS="prophet-mod-v1 ./train.csv ./test.csv"

# Create virtual environment
echo "Creating virtual environment: $ENV_NAME"
python3.9 -m venv $ENV_NAME

# Activate virtual environment
echo "Activating environment: $ENV_NAME"
source $ENV_NAME/bin/activate

# Install requirements
echo "Installing requirements from $REQUIREMENTS_PATH"
pip install -r $REQUIREMENTS_PATH

# Run the Python script
echo "Running Python script"
python $PYTHON_SCRIPT $PYTHON_ARGS

# Deactivate virtual environment
echo "Deactivating environment"
deactivate