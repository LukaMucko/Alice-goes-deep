import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy

class MultiViewLightningModule(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=2e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.weight_decay = 1e-4
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch 
        
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.accuracy(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        val_loss = F.cross_entropy(logits, y)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", self.accuracy(logits, y), prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Initial restart interval
            T_mult=2,  # Multiply interval by 2 after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

