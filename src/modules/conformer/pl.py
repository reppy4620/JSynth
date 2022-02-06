import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from pytorch_lightning import LightningModule

from .model import ConformerModel
from .dataset import ConformerDataset, collate_fn
from ..common.schedulers import Scheduler
from ..utils import add_prefix


class ConformerModule(LightningModule):
    def __init__(self, params):
        super(ConformerModule, self).__init__()
        self.params = params
        self.model = ConformerModel(params.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss_dict = self.model.compute_loss(batch)
        loss = loss_dict['loss']
        loss_dict = add_prefix(loss_dict, 'train')
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss_dict = self.model.compute_loss(batch)
        loss_dict = add_prefix(loss_dict, 'valid')
        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        if batch_idx == 0:
            o = self.model(batch)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), **self.params.optimizer)
        scheduler = Scheduler.from_config(opt, self.params.scheduler, self.trainer.current_epoch-1)
        return [opt], [scheduler]

    def setup(self, stage=None):
        self.ds = ConformerDataset(self.params.data)

    def train_dataloader(self):
        train_ds = Subset(self.ds, list(range(self.params.data.valid_size, len(self.ds))))
        return DataLoader(
            train_ds,
            batch_size=self.params.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        valid_ds = Subset(self.ds, list(range(self.params.data.valid_size)))
        return DataLoader(
            valid_ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn
        )
