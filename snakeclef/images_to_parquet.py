import argparse
import os
from pathlib import Path

from pyspark.sql import Row
from pyspark.sql.functions import element_at, regexp_replace, split

from snakeclef.utils import get_spark

"""
Before running this script, make sure you have downloaded and extracted the dataset into the data folder.
Use the bash file `download_extract_dataset.sh`
"""


# Image dataframe
def create_image_df(spark, base_dir: Path):
    # Load all files from the base directory as binary data
    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(base_dir.as_posix())
    )

    # Construct the string to be replaced - adjust this based on your actual base path
    to_remove = "file:" + str(base_dir.parents[0])

    # Remove "file:{base_dir.parents[0]" from path column
    image_df = image_df.withColumn("path", regexp_replace("path", to_remove, ""))

    # Split the path into an array of elements
    split_path = split(image_df["path"], "/")

    # Extract metadata from the file path
    image_final_df = (
        image_df.withColumn("folder_name", element_at(split_path, -4))
        .withColumn("year", element_at(split_path, -3))
        .withColumn("binomial_name", element_at(split_path, -2))
        .withColumn("file_name", element_at(split_path, -1))
    )

    # Select and rename columns to fit the target schema, including renaming 'content' to 'image_binary_data'
    image_final_df = image_final_df.select(
        "path",
        "folder_name",
        "year",
        "binomial_name",
        "file_name",
        image_final_df["content"].alias("data"),
    )

    # Create a new column "image_path" by removing "/SnakeCLEF2023-small_size/" from "path"
    # This column will be used to join with the metadata df later
    image_final_df = image_final_df.withColumn(
        "image_path", regexp_replace("path", f"^/{base_dir.parts[-1]}/", "")
    )
    return image_final_df


def create_metadata_df(spark, raw_root: str, meta_dataset_name: str, dataset_name: str):
    # Read the iNaturalist metadata CSV file
    meta_df = spark.read.csv(
        f"{raw_root}/{meta_dataset_name}.csv",
        header=True,
        inferSchema=True,
    )

    # Cache the DataFrame to optimize subsequent operations
    meta_df.cache()

    # Drop duplicate entries based on 'image_path' before the join
    meta_df = meta_df.dropDuplicates(["image_path"])

    # Assuming you want to process image paths in a similar manner
    train_root = Path(f"/mnt/data/{dataset_name}")
    paths = sorted([p.relative_to(train_root) for p in train_root.glob("**/*.jpg")])

    # Create a DataFrame from the paths
    path_df = spark.createDataFrame([Row(path=p.as_posix()) for p in paths])

    # Let's join the metadata DataFrame with the paths DataFrame
    joined_meta_df = meta_df.join(path_df, meta_df.image_path == path_df.path, "inner")

    # Dropping columns to avoid confusion
    joined_meta_df = joined_meta_df.drop("path", "binomial_name")
    return joined_meta_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for SnakeCLEF2023."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(os.getcwd())),
        help="Base directory path for image data",
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="gs://dsgt-clef-snakeclef-2024/raw/",
        help="Root directory path for metadata",
    )
    parser.add_argument(
        "--gcs_output_path",
        type=str,
        default="gs://dsgt-clef-snakeclef-2024/data/parquet_files/image_data",
        help="GCS path for output Parquet files",
    )

    return parser.parse_args()


def main():
    """Main function that processes data and writes the output dataframe to GCS"""
    args = parse_args()

    # Initialize Spark
    spark = get_spark()

    # Convert base_dir_path to a Path object here
    dataset_name = "SnakeCLEF2023-small_size"
    base_dir = Path(args.base_dir)
    base_dir = base_dir.parents[0] / "data" / dataset_name
    raw_root = args.raw_root
    meta_dataset_name = "SnakeCLEF2023-TrainMetadata-iNat"

    # Create image dataframe
    image_df = create_image_df(spark, base_dir=base_dir)

    # Create metadata dataframe
    metadata_df = create_metadata_df(
        spark,
        raw_root=raw_root,
        meta_dataset_name=meta_dataset_name,
        dataset_name=dataset_name,
    )

    # Perform an inner join on the 'image_path' column
    final_df = image_df.join(metadata_df, "image_path", "inner")

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(args.gcs_output_path)


if __name__ == "__main__":
    main()
