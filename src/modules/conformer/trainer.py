from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from accelerate import Accelerator
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .module import ConformerModule
from .utils import Tracker
from ..common.schedulers.noam import NoamLR


class Trainer:
    def __init__(self, config_path):
        self.config_path = config_path

    def run(self):
        config = OmegaConf.load(self.config_path)

        accelerator = Accelerator(fp16=False)

        seed_everything(config.seed)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        OmegaConf.save(config, output_dir / 'config.yaml')

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=f'{str(output_dir)}/logs')
        else:
            writer = None

        module = ConformerModule(config)
        train_loader, valid_loader = module.configure_dataloaders()

        model = module.model
        optimizer = optim.AdamW(model.parameters(), eps=1e-9, **config.optimizer)

        epochs = self.load(config, model, optimizer)

        model, optimizer, train_loader, valid_loader = accelerator.prepare(
            model, optimizer, train_loader, valid_loader
        )
        scheduler = NoamLR(optimizer, **config.scheduler, last_epoch=epochs * len(train_loader) - 1)

        for epoch in range(epochs, config.train.num_epochs):
            self.train_step(epoch, model, optimizer, scheduler, train_loader, writer, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self.valid_step(epoch, model, valid_loader, writer)
                if (epoch + 1) % config.train.save_interval == 0:
                    self.save(
                        output_dir / 'latest.ckpt',
                        epoch,
                        (epoch+1)*len(train_loader),
                        accelerator.unwrap_model(model),
                        optimizer
                    )
        if accelerator.is_main_process:
            writer.close()

    def train_step(self, epoch, model, optimizer, scheduler, loader, writer, accelerator):
        model.train()
        tracker = Tracker()
        bar = tqdm(desc=f'Epoch: {epoch + 1}', total=len(loader), disable=not accelerator.is_main_process)
        for i, batch in enumerate(loader):
            loss_dict = model.compute_loss(batch)
            optimizer.zero_grad()
            accelerator.backward(loss_dict['loss'])
            accelerator.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            bar.update()
            loss_dict = {k: l.item() for k, l in loss_dict.items()}
            tracker.update(**loss_dict)
            self.set_losses(bar, tracker)
        self.set_losses(bar, tracker)
        if accelerator.is_main_process:
            self.write_losses(epoch, writer, tracker, mode='train')
        bar.close()

    def valid_step(self, epoch, model, loader, writer):
        model.eval()
        tracker = Tracker()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                loss_dict = model.compute_loss(batch)
                loss_dict = {k: l.item() for k, l in loss_dict.items()}
                tracker.update(**loss_dict)
                if i == 0:
                    (
                        *labels,
                        x_length,
                        y,
                        y_length,
                        duration,
                        pitch,
                        energy
                    ) = batch
                    o = model([*labels, x_length])
                    plt.figure(figsize=(8, 6))
                    plt.subplot(2, 1, 1)
                    plt.imshow(o[0].squeeze().detach().cpu().numpy(), aspect='auto', origin='lower')
                    plt.subplot(2, 1, 2)
                    plt.imshow(y[0].squeeze().detach().cpu().numpy(), aspect='auto', origin='lower')
                    plt.savefig(f'./out/latest.png')
                    plt.close()
        self.write_losses(epoch, writer, tracker, mode='valid')

    def prepare_data(self, config):
        data_dir = Path(config.data_dir)
        assert data_dir.exists()

        fns = list(sorted(list(data_dir.glob('*.pt'))))
        valid_size = 100
        valid = fns[:valid_size]
        train = fns[valid_size:]
        return train, valid

    def load(self, config, model, optimizer):
        if config.resume_checkpoint:
            checkpoint = torch.load(f'{config.model_dir}/latest.ckpt')
            epochs = checkpoint['epoch']
            iteration = checkpoint['iteration']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded {iteration}iter model and optimizer.')
            return epochs + 1
        else:
            return 0

    def save(self, save_path, epoch, iteration, model, optimizer):
        torch.save({
            'epoch': epoch,
            'iteration': iteration,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_path)

    def write_losses(self, epoch, writer, tracker, mode='train'):
        for k, v in tracker.items():
            writer.add_scalar(f'{mode}/{k}', v.mean(), epoch)

    def set_losses(self, bar, tracker):
        bar.set_postfix_str(f', '.join([f'{k}: {v.mean():.6f}' for k, v in tracker.items()]))
