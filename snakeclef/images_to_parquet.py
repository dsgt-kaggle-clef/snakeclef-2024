import argparse
import os
from pathlib import Path

from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, regexp_extract, regexp_replace

from snakeclef.utils import get_spark


# Image dataframe
def create_image_df(base_dir):
    # Load all files from the base directory as binary data
    # Convert Path object to string when passing to PySpark
    image_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(str(base_dir))
    )

    # Construct the string to be replaced - adjust this based on your actual base path
    to_remove = "file:" + str(base_dir.parents[0])

    # Extract metadata from the file path
    image_final_df = (
        image_df.withColumn("path", regexp_replace("path", to_remove, ""))
        .withColumn("folder_name", lit("SnakeCLEF2023-small_size"))
        .withColumn("year", regexp_extract("path", ".*/(\\d{4})/.*", 1))
        .withColumn("binomial_name", regexp_extract("path", ".*/(\\d{4})/(.*)/.*", 2))
        .withColumn("file_name", regexp_extract("path", ".*/([^/]+)$", 1))
    )

    # Select and rename columns to fit the target schema, including renaming 'content' to 'image_binary_data'
    image_final_df = image_final_df.select(
        "path",
        "folder_name",
        "year",
        "binomial_name",
        "file_name",
        image_final_df["content"].alias("image_binary_data"),
    )

    # Create a new column "image_path" by removing "/SnakeCLEF2023-small_size/" from "path"
    image_final_df = image_final_df.withColumn(
        "image_path", regexp_replace("path", "^/SnakeCLEF2023-small_size/", "")
    )

    # Print Schema
    return image_final_df


def create_metadata_df(raw_root: str):
    # Metadata dataframes
    train_meta_hm = spark.read.csv(
        f"{raw_root}/SnakeCLEF2023-TrainMetadata-HM.csv", header=True, inferSchema=True
    )
    train_meta_inat = spark.read.csv(
        f"{raw_root}/SnakeCLEF2023-TrainMetadata-iNat.csv",
        header=True,
        inferSchema=True,
    )

    # Metadata dataframe
    meta_df = (
        (
            # make this table consistent with the inat one
            train_meta_hm.withColumn(
                "observation_id", F.split("observation_id", " ")[1].cast("int")
            ).select(train_meta_inat.columns)
        )
        .union(train_meta_inat)
        .dropDuplicates()
        .repartition(1)
    ).cache()

    # let's grab all the paths
    train_root = Path("/mnt/data/SnakeCLEF2023-small_size")
    paths = sorted([p.relative_to(train_root) for p in train_root.glob("**/*.jpg")])

    # Create path dataframe
    path_df = spark.createDataFrame([Row(path=p.as_posix()) for p in paths])

    def remove_leading_parent(col):
        """remove the leading parent directory from the path
        e.g. 1992/Lampropeltis_annulata/70994554.jpg turns into Lampropeltis_annulata/70994554.jpg
        """
        return F.regexp_replace(col, "^(.+?)\/", "")

    # let's join the two tables
    joined_meta_df = meta_df.withColumn(
        "path", remove_leading_parent("image_path")
    ).join(
        path_df.withColumn("path", remove_leading_parent("path")),
        on="path",
        how="right",
    )

    # Renaming columns in joined_meta_df before the join to avoid confusion
    joined_meta_df = joined_meta_df.withColumnRenamed(
        "path", "meta_path"
    ).withColumnRenamed("binomial_name", "meta_binomial_name")
    return joined_meta_df


def main(base_dir_path, raw_root_path, gcs_output_path):
    """Main function that processes data and writes the result to GCS"""
    base_dir = Path(base_dir_path)  # Convert base_dir_path to a Path object here
    raw_root = raw_root_path

    # Create DataFrames
    image_df = create_image_df(base_dir=base_dir)
    metadata_df = create_metadata_df(raw_root=raw_root)

    # Perform an inner join on the 'image_path' column
    joined_df = image_df.join(metadata_df, "image_path", "inner")

    # Now, if you wish to drop the renamed columns from joined_meta_df:
    final_df = joined_df.drop("meta_path", "meta_binomial_name")

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(gcs_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process images and metadata for SnakeCLEF2023."
    )
    parser.add_argument(
        "--base_dir", type=str, required=True, help="Base directory path for image data"
    )
    parser.add_argument(
        "--raw_root", type=str, required=True, help="Root directory path for metadata"
    )
    parser.add_argument(
        "--gcs_output_path",
        type=str,
        required=True,
        help="GCS path for output Parquet files",
    )

    args = parser.parse_args()

    # Adjust the default base_dir and raw_root if not provided
    if not args.base_dir:
        curr_dir = Path(os.getcwd())
        args.base_dir = str(curr_dir.parents[1] / "data" / "SnakeCLEF2023-small_size")

    if not args.raw_root:
        args.raw_root = "gs://dsgt-clef-snakeclef-2024/raw/"

    if not args.gcs_output_path:
        args.gcs_output_path = (
            "gs://dsgt-clef-snakeclef-2024/data/parquet_files/image_data"
        )

    # Call the main function with the processed arguments
    main(
        base_dir_path=args.base_dir,
        raw_root_path=args.raw_root,
        gcs_output_path=args.gcs_output_path,
    )
