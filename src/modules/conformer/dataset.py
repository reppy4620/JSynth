from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..common.tokenizers import TTSLearnTokenizer


class ConformerDataset(Dataset):
    def __init__(self, params):
        super().__init__()
        self.data = list(sorted(Path(params.data_dir).glob('*.pt')))
        self.tokenizer = TTSLearnTokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (
            _,
            mel,
            label,
            duration,
            pitch,
            energy
        ) = torch.load(self.data[idx])
        phoneme, is_extra = self.tokenizer(label)
        return (
            mel.transpose(-1, -2),
            phoneme,
            is_extra.view(-1, 1),
            duration.transpose(-1, -2),
            pitch.transpose(-1, -2),
            energy.transpose(-1, -2)
        )


def collate_fn(batch):
    (
        mel,
        phoneme,
        is_extra,
        duration,
        pitch,
        energy
    ) = tuple(zip(*batch))

    x_length = torch.LongTensor([len(x) for x in phoneme])
    phoneme = pad_sequence(phoneme, batch_first=True)
    is_extra = pad_sequence(is_extra, batch_first=True).transpose(-1, -2)

    y_length = torch.LongTensor([x.size(0) for x in mel])
    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)

    pitch = pad_sequence(pitch, batch_first=True).transpose(-1, -2)
    energy = pad_sequence(energy, batch_first=True).transpose(-1, -2)
    duration = pad_sequence(duration, batch_first=True).transpose(-1, -2)

    return (
        phoneme,
        is_extra,
        x_length,
        mel,
        y_length,
        duration,
        pitch,
        energy
    )
