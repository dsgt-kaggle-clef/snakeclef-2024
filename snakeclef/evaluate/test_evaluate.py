import numpy as np
import pandas as pd
import PIL
import pytest
import torch
from pytorch_lightning import Trainer

from .data import ImageDataset, InferenceDataModel
from .model import LinearClassifier
from .submission import make_submission


class TestingInferenceDataModel(InferenceDataModel):
    def train_dataloader(self):
        for batch in self.predict_dataloader():
            # add a label to the batch with classes from 0 to 9
            batch["label"] = torch.randint(0, 10, (batch["features"].shape[0],))
            yield batch


@pytest.fixture
def images_root(tmp_path):
    images_root = tmp_path / "images"
    images_root.mkdir()
    for i in range(10):
        img = PIL.Image.fromarray(
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        )
        img.save(images_root / f"{i}.jpg")
    return images_root


@pytest.fixture
def metadata(tmp_path, images_root):
    res = []
    for i, img in enumerate(images_root.glob("*.jpg")):
        res.append({"image_path": img.name, "observation_id": i})
    df = pd.DataFrame(res)
    df.to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path / "metadata.csv"


@pytest.fixture
def model_checkpoint(tmp_path, metadata, images_root):
    model_checkpoint = tmp_path / "model.ckpt"
    model = LinearClassifier(768, 10)
    trainer = Trainer(max_epochs=1, fast_dev_run=True)
    dm = TestingInferenceDataModel(metadata, images_root)
    trainer.fit(model, dm)
    trainer.save_checkpoint(model_checkpoint)
    return model_checkpoint


def test_image_dataset(images_root, metadata):
    dataset = ImageDataset(metadata, images_root)
    assert len(dataset) == 10
    for i in range(10):
        assert dataset[i]["features"].shape == torch.Size([3, 100, 100])


def test_inference_datamodel(images_root, metadata):
    batch_size = 5
    model = InferenceDataModel(metadata, images_root, batch_size=batch_size)
    model.setup()
    assert len(model.dataloader) == 2
    for batch in model.predict_dataloader():
        assert set(batch.keys()) == {"features", "observation_id"}
        assert batch["features"].shape == torch.Size([batch_size, 768])


def test_model_checkpoint(model_checkpoint):
    model = LinearClassifier.load_from_checkpoint(model_checkpoint)
    assert model


def test_make_submission(model_checkpoint, metadata, images_root, tmp_path):
    output_csv_path = tmp_path / "submission.csv"
    make_submission(metadata, model_checkpoint, output_csv_path, images_root)
    submission_df = pd.read_csv(output_csv_pathgit)
    assert len(submission_df) == 10
    assert set(submission_df.columns) == {"observation_id", "class_id"}
    assert submission_df["class_id"].isin(range(10)).all()
