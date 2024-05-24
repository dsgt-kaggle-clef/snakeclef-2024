from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


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
        for batch in self.dataloader:
            yield batch
