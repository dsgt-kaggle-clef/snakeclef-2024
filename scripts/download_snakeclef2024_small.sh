#!/bin/bash

# This script downloads and extracts the SnakeCLEF2023 dataset.

set -e # Exit immediately if a command exits with a non-zero status.

DATASET_URL="gs://dsgt-clef-snakeclef-2024/raw/SnakeCLEF2023-train-small_size.tar.gz"
DESTINATION_DIR="/mnt/data"
DATASET_NAME="SnakeCLEF2023-train-small_size.tar.gz"
DESTINATION_PATH="$DESTINATION_DIR/$DATASET_NAME"

# Prepare the destination directory
sudo mount "$DESTINATION_DIR" || true # Proceed even if mount fails, assuming it's already mounted
sudo chmod -R 777 "$DESTINATION_DIR"
echo "Permissions set for $DESTINATION_DIR."

# Download the dataset (if not already downloaded)
echo "Downloading dataset to $DESTINATION_DIR..."
gcloud storage cp "$DATASET_URL" "$DESTINATION_DIR" || {
    echo "Failed to download the dataset."
    exit 1
}

# Extract the dataset
echo "Extracting dataset..."
tar -xzf "$DESTINATION_PATH" -C "$DESTINATION_DIR"
echo "Dataset extracted to $DESTINATION_DIR."

# Final listing and disk usage report
echo "Final contents of $DESTINATION_DIR:"
ls "$DESTINATION_DIR"
echo "Disk usage and free space:"
df -h

echo "Script completed successfully."
