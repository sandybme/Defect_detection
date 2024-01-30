#!/bin/bash

# Create a Conda environment
conda create -n deepl python=3.10 -y
source activate deepl

# Install necessary libraries from requirements.txt
conda install pytorch torchvision -c pytorch
echo "Environment setup complete."
