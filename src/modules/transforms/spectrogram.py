from .base import TransformBase
from .utils import spectrogram_torch
from .utils import mel_spectrogram


class Spectrogram(TransformBase):
    def transform(self, x):
        return spectrogram_torch(x)


class MelSpectrogram(TransformBase):
    def transform(self, x):
        return mel_spectrogram(x)[0]


class MelSpectrogramWithEnergy(TransformBase):
    def transform(self, x):
        return mel_spectrogram(x)
