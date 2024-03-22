import os
import sys
from contextlib import contextmanager

from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark(
    cores=4, memory="8g", local_dir="/mnt/data/tmp", gpu_resources=False, **kwargs
):
    """Get a spark session for a single driver."""
    builder = (
        SparkSession.builder.config("spark.driver.memory", memory)
        .config("spark.driver.cores", cores)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.local.dir", local_dir)
    )

    if gpu_resources:
        builder = (
            builder.config("spark.rapids.memory.pinnedPool.size", "2G")
            .config("spark.executor.resource.gpu.amount", "1")
            .config("spark.task.resource.gpu.amount", "0.1")
            .config(
                "spark.executor.resource.gpu.discoveryScript", "getGpusResources.sh"
            )
            .config("spark.rapids.sql.concurrentGpuTasks", "1")
        )  # Adjust based on your GPU's capability

    for k, v in kwargs.items():
        builder = builder.config(k, v)
    return builder.getOrCreate()


@contextmanager
def spark_resource(*args, **kwargs):
    """A context manager for a spark session."""
    spark = None
    try:
        spark = get_spark(*args, **kwargs)
        yield spark
    finally:
        if spark is not None:
            spark.stop()
