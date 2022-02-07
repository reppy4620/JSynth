import torch.nn as nn

from abc import abstractmethod


class ModuleBase(nn.Module):
    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def validating_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def configure_dataloaders(self):
        pass

    @abstractmethod
    def configure_optimizers(self, epochs):
        pass