import torch.nn as nn
from abc import abstractmethod


class ModelBase(nn.Module):
    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        pass
