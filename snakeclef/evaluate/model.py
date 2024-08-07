import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)


class LinearClassifier(pl.LightningModule):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.save_hyperparameters()  # Saves hyperparams in the checkpoints
        self.model = nn.Linear(num_features, num_classes)
        self.learning_rate = 0.002
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average="weighted")
        self.f1_score = MulticlassF1Score(num_classes=num_classes, average="weighted")
        self.precision = MulticlassPrecision(
            num_classes=num_classes, average="weighted"
        )
        self.recall = MulticlassRecall(num_classes=num_classes, average="weighted")

    def forward(self, x):
        return torch.log_softmax(self.model(x), dim=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _run_step(self, batch, batch_idx, step_name):
        x, y = batch["features"], batch["label"]
        logits = self(x)
        loss = torch.nn.functional.nll_loss(logits, y)
        self.log(f"{step_name}_loss", loss, prog_bar=True)
        self.log(
            f"{step_name}_accuracy",
            self.accuracy(logits, y),
            on_step=False,
            on_epoch=True,
        )
        if step_name != "train":
            self.log(
                f"{step_name}_f1",
                self.f1_score(logits, y),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step_name}_precision",
                self.precision(logits, y),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{step_name}_recall",
                self.recall(logits, y),
                on_step=False,
                on_epoch=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._run_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits = self(batch["features"])
        return {
            "logits": logits,
            "class_id": torch.argmax(logits, dim=1),
            "observation_id": batch["observation_id"],
        }
