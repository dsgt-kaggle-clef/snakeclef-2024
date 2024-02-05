from snakeclef.utils import get_spark

if __name__ == "__main__":
    spark = get_spark()
    raw_root = "gs://dsgt-clef-snakeclef-2024/raw/"
    meta = spark.read.csv(
        f"{raw_root}/SnakeCLEF2023-TrainMetadata-iNat.csv",
        header=True,
        inferSchema=True,
    )
    meta.printSchema()
    meta.show(3, vertical=True, truncate=80)
