import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from .model import ConformerModel
from ..common.model import ModuleBase
from ..common.datasets import TTSDataset
from ..common.schedulers import Scheduler
from ..utils import add_prefix


class ConformerModule(ModuleBase):
    def __init__(self, params):
        super(ConformerModule, self).__init__()
        self.params = params
        self.model = ConformerModel(params.model)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss_dict = self.model.compute_loss(batch)
        loss_dict = add_prefix(loss_dict, 'train')
        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss_dict = self.model.compute_loss(batch)
        loss_dict = add_prefix(loss_dict, 'valid')
        if batch_idx == 0:
            (
                *labels,
                x_length,
                y,
                y_length,
                duration,
                pitch,
                energy
            ) = batch
            o = self.model([*labels, x_length])
            plt.figure(figsize=(8, 6))
            plt.subplot(2, 1, 1)
            plt.imshow(o[0].squeeze().detach().cpu().numpy(), aspect='auto', origin='lower')
            plt.subplot(2, 1, 2)
            plt.imshow(y[0].squeeze().detach().cpu().numpy(), aspect='auto', origin='lower')
            plt.savefig(f'{self.params.output_dir}/latest.png')
            plt.close()
        return loss_dict

    def configure_optimizers(self, epochs):
        opt = optim.AdamW(self.model.parameters(), eps=1e-9, **self.params.optimizer)
        scheduler = Scheduler.from_config(opt, self.params.scheduler, epochs-1)
        self.optimizer = opt
        return opt, scheduler

    def configure_dataloaders(self):
        Dataset, collate_fn = TTSDataset.from_config(self.params.data)
        ds = Dataset(self.params.data)
        train_ds = Subset(ds, list(range(self.params.data.valid_size, len(ds))))
        valid_ds = Subset(ds, list(range(self.params.data.valid_size)))
        train_loader = DataLoader(
            train_ds,
            batch_size=self.params.train.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_fn
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn
        )
        return train_loader, valid_loader
