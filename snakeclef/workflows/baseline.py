from argparse import ArgumentParser

import luigi
import luigi.contrib.gcs
from pyspark.ml import Pipeline
from pyspark.ml.feature import SQLTransformer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from snakeclef.transforms import DCTN, ExtractCLSToken, WrappedDinoV2
from snakeclef.utils import spark_resource

from .classifier import TrainDCTEmbeddingClassifier


class ProcessBase(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    should_subset = luigi.BoolParameter(default=False)
    sample_col = luigi.Parameter(default="observation_id")
    num_partitions = luigi.OptionalIntParameter(default=32)
    sample_id = luigi.OptionalIntParameter(default=None)
    num_sample_id = luigi.OptionalIntParameter(default=10)

    def output(self):
        if self.sample_id is None:
            # save both the model pipeline and the dataset
            return luigi.contrib.gcs.GCSTarget(f"{self.output_path}/data/_SUCCESS")
        else:
            return luigi.contrib.gcs.GCSTarget(
                f"{self.output_path}/data/sample_id={self.sample_id}/_SUCCESS"
            )

    @property
    def feature_columns(self) -> list:
        raise NotImplementedError()

    def pipeline(self) -> Pipeline:
        raise NotImplementedError()

    def transform(self, model, df, features) -> DataFrame:
        transformed = model.transform(df)

        if self.sample_id is not None:
            transformed = (
                transformed.withColumn(
                    "sample_id",
                    F.crc32(F.col(self.sample_col).cast("string")) % self.num_sample_id,
                )
                .where(F.col("sample_id") == self.sample_id)
                .drop("sample_id")
            )

        for c in features:
            # check if the feature is a vector and convert it to an array
            if "array" in transformed.schema[c].simpleString():
                continue
            transformed = transformed.withColumn(c, vector_to_array(F.col(c)))
        return transformed

    def _get_subset(self, df):
        # Get subset of images to test pipeline
        subset_df = (
            df.where(F.col(self.sample_col).isin([5156122, 86508209]))  # top 2 species
            .orderBy(F.rand(1000))
            .limit(200)
            .cache()
        )
        return subset_df

    def run(self):
        with spark_resource(
            **{"spark.sql.shuffle.partitions": self.num_partitions}
        ) as spark:
            df = spark.read.parquet(self.input_path)

            if self.should_subset:
                # Get subset of data to test pipeline
                df = self._get_subset(df=df)

            model = self.pipeline().fit(df)
            model.write().overwrite().save(f"{self.output_path}/model")
            transformed = self.transform(model, df, self.feature_columns)

            if self.sample_id is None:
                output_path = f"{self.output_path}/data"
            else:
                output_path = f"{self.output_path}/data/sample_id={self.sample_id}"

            transformed.repartition(self.num_partitions).write.mode(
                "overwrite"
            ).parquet(output_path)


class ProcessDino(ProcessBase):
    sql_statement = luigi.Parameter()

    @property
    def feature_columns(self) -> list:
        return ["dino_embedding"]

    def pipeline(self):
        dino = WrappedDinoV2(input_col="data", output_col="dino_embedding")
        return Pipeline(stages=[dino, SQLTransformer(statement=self.sql_statement)])


class ProcessDCT(ProcessBase):
    sql_statement = luigi.Parameter()

    @property
    def feature_columns(self) -> list:
        return ["dct_embedding"]

    def pipeline(self):
        dct = DCTN(input_col="dino_embedding", output_col="dct_embedding")
        return Pipeline(stages=[dct, SQLTransformer(statement=self.sql_statement)])


class ProcessCLS(ProcessBase):
    sql_statement = luigi.Parameter()

    @property
    def feature_columns(self) -> list:
        return ["cls_embedding"]

    def pipeline(self):
        cls_extractor = ExtractCLSToken(
            input_col="dino_embedding", output_col="cls_embedding"
        )
        return Pipeline(
            stages=[cls_extractor, SQLTransformer(statement=self.sql_statement)]
        )


class Workflow(luigi.Task):
    input_path = luigi.Parameter()
    output_path = luigi.Parameter()
    default_root_dir = luigi.Parameter()
    process_test_data = luigi.OptionalBoolParameter(default=False)
    use_cls_token = luigi.OptionalBoolParameter(default=False)

    def run(self):
        # training workflow parameters
        subset_list = [True, False]
        train_model = True
        sample_col = "observation_id"
        dino_sql_statement = "SELECT path, observation_id, dino_embedding FROM __THIS__"
        dct_sql_statement = "SELECT path, observation_id, dct_embedding FROM __THIS__"
        cls_sql_statement = "SELECT path, observation_id, cls_embedding FROM __THIS__"

        # test workflow parameters
        if self.process_test_data:
            subset_list = [False]  # Skip subset for test data
            train_model = False  # Skip model training
            sample_col = "path"
            dino_sql_statement = "SELECT path, dino_embedding FROM __THIS__"
            dct_sql_statement = "SELECT path, dct_embedding FROM __THIS__"
            cls_sql_statement = "SELECT path, cls_embedding FROM __THIS__"

        # Run jobs with subset and full-size data
        for subset in subset_list:
            final_output_path = self.output_path
            if subset:
                subset_path = f"subset_{self.output_path.split('/')[-1]}"
                final_output_path = self.output_path.replace(
                    self.output_path.split("/")[-1], subset_path
                )
            yield ProcessDino(
                input_path=self.input_path,
                output_path=f"{final_output_path}/dino",
                should_subset=subset,
                # sample_id=i,
                # num_sample_id=10,
                sample_col=sample_col,
                sql_statement=dino_sql_statement,
            )

            yield ProcessDCT(
                input_path=f"{final_output_path}/dino/data",
                output_path=f"{final_output_path}/dino_dct",
                should_subset=subset,
                sample_col=sample_col,
                sql_statement=dct_sql_statement,
            )
            yield ProcessCLS(
                input_path=f"{final_output_path}/dino/data",
                output_path=f"{final_output_path}/dino_cls_token",
                should_subset=subset,
                sample_col=sample_col,
                sql_statement=cls_sql_statement,
            )

        # Train classifier outside of the subset loop
        if train_model:
            for limit_species in [None]:
                # use the Dino-DCT dataset for training the classifier
                data_path = "dino_dct/data"
                input_path = f"{self.output_path}/{data_path}"
                feature_col = "dct_embedding"
                final_default_dir = self.default_root_dir
                # use the CLS token dataset for training the classifier
                if self.use_cls_token:
                    data_path = "dino_cls_token/data"
                    input_path = f"{self.output_path}/{data_path}"
                    feature_col = "cls_embedding"
                    final_default_dir = f"{final_default_dir}-cls-token"
                if limit_species:
                    final_default_dir = (
                        f"{final_default_dir}-limit-species-{limit_species}"
                    )
                print(f"\ninput_path: {input_path}")
                print(f"feature_col: {feature_col}")
                print(f"default_root: {final_default_dir}\n")
                yield TrainDCTEmbeddingClassifier(
                    input_path=input_path,
                    feature_col=feature_col,
                    default_root_dir=final_default_dir,
                    limit_species=limit_species,
                )


def parse_args():
    parser = ArgumentParser(description="Luigi pipeline")
    parser.add_argument(
        "--gcs-root-path",
        type=str,
        default="gs://dsgt-clef-snakeclef-2024",
        help="Root directory for snakeclef-2024 in GCS",
    )
    parser.add_argument(
        "--train-data-path",
        type=str,
        default="data/parquet_files/SnakeCLEF2023-train-small_size",
        help="Root directory for training data in GCS",
    )
    parser.add_argument(
        "--output-name-path",
        type=str,
        default="data/process/training_small_v1",
        help="GCS path for output Parquet files",
    )
    parser.add_argument(
        "--model-dir-path",
        type=str,
        default="models/torch-petastorm-v1",
        help="Default root directory for storing the pytorch classifier runs",
    )
    parser.add_argument(
        "--process-test-data",
        type=bool,
        default=False,
        help="If True, set pipeline to process the test data and extract embeddings",
    )
    parser.add_argument(
        "--use-cls-token",
        type=bool,
        default=False,
        help="If True, use the CLS token from the DINOv2 ViT model for classification",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Input and output paths for training workflow
    input_path = f"{args.gcs_root_path}/{args.train_data_path}"
    output_path = f"{args.gcs_root_path}/{args.output_name_path}"
    default_root_dir = f"{args.gcs_root_path}/{args.model_dir_path}"
    process_test_data = args.process_test_data
    use_cls_token = args.use_cls_token

    # update workflow parameters for processing test data
    if process_test_data:
        input_path = f"{args.gcs_root_path}/data/parquet_files/PlantCLEF2024_test"
        output_path = f"{args.gcs_root_path}/data/process/test_v1"

    luigi.build(
        [
            Workflow(
                input_path=input_path,
                output_path=output_path,
                default_root_dir=default_root_dir,
                process_test_data=process_test_data,
                use_cls_token=use_cls_token,
            )
        ],
        scheduler_host="services.us-central1-a.c.dsgt-clef-2024.internal",
    )
