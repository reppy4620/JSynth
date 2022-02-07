from pathlib import Path

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import Tracker
from ..pl_module import PLModule


class Trainer:
    def __init__(self,
                 config,
                 resume_checkpoint=None):
        self.config = config
        self.resume_checkpoint = resume_checkpoint

    def run(self):
        config = self.config

        accelerator = Accelerator(fp16=False)

        seed_everything(config.seed)

        output_dir = Path(config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        OmegaConf.save(config, output_dir / 'config.yaml')

        if accelerator.is_main_process:
            writer = SummaryWriter(log_dir=f'{str(output_dir)}/logs')
        else:
            writer = None

        module = PLModule.from_config(config)
        train_loader, valid_loader = module.configure_dataloaders()

        epochs = 1
        if self.resume_checkpoint:
            epochs = self.load(config, module)

        optimizer, scheduler = module.configure_optimizers(epochs=epochs-1)

        module, optimizer, train_loader, valid_loader = accelerator.prepare(
            module, optimizer, train_loader, valid_loader
        )

        for epoch in range(epochs, config.train.num_epochs+1):
            self.train_step(epoch, module, optimizer, scheduler, train_loader, writer, accelerator)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                self.valid_step(epoch, module, valid_loader, writer)
                self.save(
                    output_dir / 'last.ckpt',
                    epoch,
                    accelerator.unwrap_model(module)
                )
        if accelerator.is_main_process:
            writer.close()

    def train_step(self, epoch, module, optimizer, scheduler, loader, writer, accelerator):
        module.train()
        tracker = Tracker()
        bar = tqdm(desc=f'Epoch: {epoch + 1}', total=len(loader), disable=not accelerator.is_main_process)
        for i, batch in enumerate(loader):
            loss_dict = module.training_step(batch, i)
            optimizer.zero_grad()
            accelerator.backward(loss_dict['loss'])
            accelerator.clip_grad_norm_(module.parameters(), 5)
            optimizer.step()
            scheduler.step()
            bar.update()
            loss_dict = {k: l.item() for k, l in loss_dict.items()}
            tracker.update(**loss_dict)
            self.set_losses(bar, tracker)
        if accelerator.is_main_process:
            self.write_losses(epoch, writer, tracker, mode='train')
        bar.close()

    def valid_step(self, epoch, module, loader, writer):
        module.eval()
        tracker = Tracker()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                loss_dict = module.validation_step(batch, i)
                loss_dict = {k: l.item() for k, l in loss_dict.items()}
                tracker.update(**loss_dict)
        self.write_losses(epoch, writer, tracker, mode='valid')

    def load(self, config, module):
        if config.resume_checkpoint:
            checkpoint = torch.load(f'{config.model_dir}/latest.ckpt')
            epochs = checkpoint['epoch']
            module.load_state_dict(checkpoint['model'])
            module.optimizer.load_state_dict(checkpoint['optimizer'])
            return epochs + 1
        else:
            return 0

    def save(self, save_path, epoch, module):
        torch.save({
            'epoch': epoch,
            'model': module.state_dict(),
            'optimizer': module.optimizer.state_dict()
        }, save_path)

    def write_losses(self, epoch, writer, tracker, mode='train'):
        for k, v in tracker.items():
            writer.add_scalar(f'{mode}/{k}', v.mean(), epoch)

    def set_losses(self, bar, tracker):
        bar.set_postfix_str(f', '.join([f'{k}: {v.mean():.6f}' for k, v in tracker.items()]))
