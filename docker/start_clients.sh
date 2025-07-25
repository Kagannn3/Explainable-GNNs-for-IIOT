#!/bin/bash

# ========================
# Script to start all clients for federated training
# using Docker Compose. Each client runs in its own
# container and executes the `train_client.py` script
# with specific arguments.
# ========================

# Set number of clients
NUM_CLIENTS=3

# Set model types to run for each client
MODELS=("lstm" "gcn" "gat")

# Set common training hyperparameters
HIDDEN_SIZE=64
LEARNING_RATE=0.001
EPOCHS=50

# Print status
echo "Starting $NUM_CLIENTS clients for each model: ${MODELS[@]}"
echo "Using hidden_size=$HIDDEN_SIZE, lr=$LEARNING_RATE, epochs=$EPOCHS"
echo

# Loop through each client (e.g., client0, client1, client2)
for CLIENT_ID in $(seq 0 $((NUM_CLIENTS - 1))); do
  for MODEL in "${MODELS[@]}"; do
    echo "Launching client $CLIENT_ID with model: $MODEL"

    # Run a Docker container for each client-model pair
    docker run -d \
      --name "client${CLIENT_ID}_${MODEL}" \
      -v "$(pwd):/app" \                             # Mount project directory into the container
      -w /app/src \                                  # Set working directory inside container
      --rm python:3.9 \                              # Use Python Docker image (or use your own image if customized)
      bash -c "pip install -r ../docker/requirements.txt && \
               python train_client.py \
               --client_id ${CLIENT_ID} \
               --model ${MODEL} \
               --hidden_size ${HIDDEN_SIZE} \
               --lr ${LEARNING_RATE} \
               --epochs ${EPOCHS}"
  done
done

echo
echo "All clients launched. Use 'docker ps' to check containers."
# Note: Ensure that the `train_client.py` script is set up to handle the command-line arguments
# and that the Docker image has all necessary dependencies installed.