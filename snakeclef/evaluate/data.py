from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from transformers import AutoImageProcessor, AutoModel


class TransformDino(v2.Transform):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, batch):
        model_inputs = self.processor(images=batch["features"], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            last_hidden_states = outputs.last_hidden_state
        # extract the cls token
        batch["features"] = last_hidden_states[:, 0]
        return batch


class ImageDataset(Dataset):
    def __init__(self, metadata_path, images_root_path):
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.images_root_path = images_root_path

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_path = Path(self.images_root_path) / row.image_path
        img = Image.open(image_path).convert("RGB")
        img = v2.ToTensor()(img)
        return {"features": img}


class InferenceDataModel(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path,
        images_root_path,
        batch_size=32,
    ):
        super().__init__()
        self.metadata_path = metadata_path
        self.images_root_path = images_root_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataloader = DataLoader(
            ImageDataset(self.metadata_path, self.images_root_path),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def predict_dataloader(self):
        transform = v2.Compose([TransformDino("facebook/dinov2-base")])
        for batch in self.dataloader:
            batch = transform(batch)
            yield batch
