from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..common.tokenizers import Tokenizer


class ConformerDataset(Dataset):
    def __init__(self, params):
        super().__init__()
        self.data = list(sorted(Path(params.data_dir).glob('*.pt')))
        self.tokenizer = Tokenizer.from_config(params.tokenizer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (
            _,
            mel,
            inputs,
            duration,
            pitch,
            energy
        ) = torch.load(self.data[idx])
        *inputs, is_transpose = self.tokenizer(inputs)
        duration = duration.float()
        non_zero_idx = duration != 0
        duration[non_zero_idx] = torch.log(duration[non_zero_idx])
        return (
            mel.transpose(-1, -2),
            *inputs,
            is_transpose,
            duration.transpose(-1, -2),
            pitch.transpose(-1, -2),
            energy.transpose(-1, -2)
        )


def collate_fn(batch):
    (
        mel,
        *inputs,
        is_transpose,
        duration,
        pitch,
        energy
    ) = tuple(zip(*batch))
    print(is_transpose)

    x_length = torch.LongTensor([len(x) for x in inputs[0]])

    inp_list = list()
    for i, inp in enumerate(inputs):
        x = pad_sequence(inp, batch_first=True)
        if is_transpose[i]:
            x = x.transpose(-1, -2)
        inp_list.append(x)

    y_length = torch.LongTensor([x.size(0) for x in mel])
    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)

    pitch = pad_sequence(pitch, batch_first=True).transpose(-1, -2)
    energy = pad_sequence(energy, batch_first=True).transpose(-1, -2)
    duration = pad_sequence(duration, batch_first=True).transpose(-1, -2)

    return (
        *inp_list,
        x_length,
        mel,
        y_length,
        duration,
        pitch,
        energy
    )
