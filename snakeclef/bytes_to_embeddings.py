import argparse
import io

import numpy as np
import torch
from PIL import Image
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql.types import ArrayType, FloatType

from .utils import get_spark


def make_predict_fn():
    """Return PredictBatchFunction"""
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

    def predict(inputs: np.ndarray) -> np.ndarray:
        images = [Image.open(io.BytesIO(input)) for input in inputs]
        model_inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**model_inputs)
            last_hidden_states = outputs.last_hidden_state

        numpy_array = last_hidden_states.cpu().numpy()
        new_shape = numpy_array.shape[:-2] + (-1,)
        numpy_array = numpy_array.reshape(new_shape)

        return numpy_array

    return predict


def embed_dataframe(df, batch_size=16):
    # batch prediction UDF
    apply_dino_pbudf = predict_batch_udf(
        make_predict_fn=make_predict_fn,
        return_type=ArrayType(FloatType()),
        batch_size=batch_size,
    )

    # add the embedding and drop the original image since we can simply join it later
    return df.withColumn("dino_embedding", apply_dino_pbudf("data")).drop("data")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process images and metadata for a dataset stored on GCS."
    )
    parser.add_argument(
        "--image-size",
        type=str,
        default="small",
        help="Size of images in dataset, from [small, medium, large]",
    )
    parser.add_argument(
        "--gcs-parquet-path",
        type=str,
        default="gs://dsgt-clef-snakeclef-2024/data/parquet_files/",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing images",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Spark
    spark = get_spark(
        # gpu_resources=True,
        memory="12g",
        **{
            "spark.sql.parquet.enableVectorizedReader": "false",
        },
    )

    input_folder = f"SnakeCLEF2023-train-{args.image_size}_size/"
    output_folder = f"DINOv2-embeddings-{args.image_size}_size/"
    input_path = args.gcs_parquet_path + input_folder
    output_path = args.gcs_parquet_path + output_folder
    print(f"Reading data from {input_path}")
    print(f"Writing data to {output_path}")

    # put processor and model as spark variables
    df = spark.read.parquet(input_path)
    df.printSchema()

    # Create image dataframe
    final_df = embed_dataframe(df, batch_size=args.batch_size)

    # Write the DataFrame to GCS in Parquet format
    final_df.write.mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    main()
