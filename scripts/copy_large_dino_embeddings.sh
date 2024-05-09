# gsutil -m cp -r gs://dsgt-clef-snakeclef-2024/data/parquet_files/DINOv2-embeddings-large_size/* gs://dsgt-clef-snakeclef-2024/data/process/subset_training_large_v1/dino/data/

#!/bin/bash

# Define the bucket and folders
BUCKET_NAME="dsgt-clef-snakeclef-2024"
SOURCE_FOLDER="data/parquet_files/DINOv2-embeddings-large_size"
DESTINATION_FOLDER="data/process/training_large_v1/dino/data"

# Execute the copy command
gsutil -m cp -r gs://${BUCKET_NAME}/${SOURCE_FOLDER}/* gs://${BUCKET_NAME}/${DESTINATION_FOLDER}/