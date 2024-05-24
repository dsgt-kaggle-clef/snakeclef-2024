import zipfile

import pandas as pd
import torch
from pytorch_lightning import Trainer

from .data import InferenceDataModel
from .model import LinearClassifier


def make_submission(
    test_metadata,
    model_path,
    output_csv_path="./submission.csv",
    images_root_path="/tmp/data/private_testset",
):
    model = LinearClassifier.load_from_checkpoint(model_path)
    dm = InferenceDataModel(
        metadata_path=test_metadata, images_root_path=images_root_path
    )
    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    predictions = trainer.predict(model, datamodule=dm)
    rows = []
    for batch in predictions:
        for observation_id, class_id in zip(batch["observation_id"], batch["class_id"]):
            row = {"observation_id": int(observation_id), "class_id": int(class_id)}
            rows.append(row)
    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    with zipfile.ZipFile("/tmp/data/private_testset.zip", "r") as zip_ref:
        zip_ref.extractall("/tmp/data")

    MODEL_PATH = "last.ckpt"
    metadata_file_path = "./SnakeCLEF2024-TestMetadata.csv"
    test_metadata = pd.read_csv(metadata_file_path)
    make_submission(test_metadata=test_metadata, model_path=MODEL_PATH)
