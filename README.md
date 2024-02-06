# snakeclef-2024

- https://www.imageclef.org/node/319

## Quickstart

Install the pre-commit hooks for formatting code:

```bash
pre-commit install
```

We are generally using a shared VM with limited space.
Install packages to the system using sudo:

```bash
sudo pip install -r requirements.txt
```

We can ignore the following message since we know what we are doing:

```
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
```

This should allow all users to use the packages without having to install them multiple times.
This is a problem with very large packages like `torch` and `spark`.

Then install the current package into your local user directory, so changes that you make are local to your own users.

```bash
pip install -e .
```

## Download and Extract Dataset Script

The `download_extract_dataset.sh` script automates the process of downloading and extracting a dataset from GCS. It allows for the customization of the dataset URL and the destination directory through command-line arguments.

### Usage

To use the script, navigate to your project directory and run the following command in your terminal:

```bash
scripts/download_extract_dataset.sh [DATASET_URL] [DESTINATION_DIR]
```

Replace **[DATASET_URL]** with the URL of the dataset you wish to download, and **[DESTINATION_DIR]** with the path where you want the dataset to be downloaded and extracted. If these arguments are not provided, the script will use its default settings.

### Example

```
scripts/download_extract_dataset.sh gs://dsgt-clef-snakeclef-2024/raw/SnakeCLEF2023-train-small_size.tar.gz /mnt/data
```

This will download the `SnakeCLEF2023-train-small_size.tar.gz` dataset from the specified Google Cloud Storage URL and extract it to `/mnt/data`.

## Create a dataframe from images and write it to GCS

The `images_to_parquet.py` script is designed to process image data and associated metadata for a specific dataset. It reads images and metadata from specified directories, performs necessary processing, and outputs the data in Parquet format to a GCS path. This script supports customizing paths for input data and output files through command-line arguments.

### Usage

Navigate to the project directory and run the script using the following command format:

```
python snakeclef/images_to_parquet.py [OPTIONS]
```

**Options:**

- `--image-root-path`: Base directory path for image data. Default is the current project directory.

- `--raw-root-path`: Root directory path for metadata. Default is `gs://dsgt-clef-snakeclef-2024/raw/`.

- `--output-path`: GCS path for output Parquet files. Default is `gs://dsgt-clef-snakeclef-2024/data/parquet_files/image_data`.

- `--dataset-name`: Dataset name downloaded from the tar file. Default is `SnakeCLEF2023-small_size`.

- `--meta-dataset-name`: Train Metadata CSV file name. Default is `SnakeCLEF2023-TrainMetadata-iNat`.

### Example Commands

Run the script with default settings:

```
python snakeclef/images_to_parquet.py
```

Run the script with custom paths:

```
python snakeclef/images_to_parquet.py --image-root-path /path/to/images --raw-root-path gs://my-custom-path/raw/ --output-path gs://my-custom-path/data/parquet_files/image_data --dataset-name MyDataset --meta-dataset-name MyMetadata
```

For detailed help on command-line options, run `python snakeclef/images_to_parquet.py --help`.
