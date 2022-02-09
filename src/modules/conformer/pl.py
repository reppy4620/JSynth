import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from pytorch_lightning import LightningModule

from .model import ConformerModel
from ..common.datasets import Dataset
from ..common.schedulers import Scheduler
from ..common.utils import add_prefix


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
            logger=True
        )

        if batch_idx == 0:
            (
                *labels,
                x_length,
                y,
                y_length
            ) = batch
            o = self.model([*labels, x_length])
            o = o[0][:, :y_length[0]]
            y = y[0][:, :y_length[0]]
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.imshow(o.detach().cpu().float().numpy(), aspect='auto', origin='lower')
            plt.subplot(2, 1, 2)
            plt.imshow(y.detach().cpu().float().numpy(), aspect='auto', origin='lower')
            plt.savefig(f'{self.params.output_dir}/latest.png')
            plt.close()

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), eps=1e-9, **self.params.optimizer)
        scheduler = Scheduler.from_config(opt, self.params.scheduler, self.trainer.current_epoch-1)
        return [opt], [scheduler]

    def setup(self, stage=None):
        self.ds, self.collate_fn = Dataset.from_config(self.params.data)

    def train_dataloader(self):
        train_ds = Subset(self.ds, list(range(self.params.data.valid_size, len(self.ds))))
        return DataLoader(
            train_ds,
            batch_size=self.params.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        valid_ds = Subset(self.ds, list(range(self.params.data.valid_size)))
        return DataLoader(
            valid_ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.collate_fn
        )
