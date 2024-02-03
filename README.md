# snakeclef-2024

- https://www.imageclef.org/node/319

## quickstart

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

This script automates the process of downloading and extracting the SnakeCLEF2023 dataset. It allows for the customization of the dataset URL and the destination directory through command-line arguments.

### Usage

To use the script, navigate to the directory where the script is located and run the following command in your terminal:

```bash
./download_extract_dataset.sh [DATASET_URL] [DESTINATION_DIR]
```

Replace [DATASET_URL] with the URL of the dataset you wish to download, and [DESTINATION_DIR] with the path where you want the dataset to be downloaded and extracted. If these arguments are not provided, the script will use its default settings.

### Example

```
./download_extract_dataset.sh gs://dsgt-clef-snakeclef-2024/raw/SnakeCLEF2023-train-small_size.tar.gz /mnt/data
```

This will download the dataset from the specified Google Cloud Storage URL and extract it to /mnt/data.
