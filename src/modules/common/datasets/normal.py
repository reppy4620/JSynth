from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from ..tokenizers import Tokenizer


class NormalDataset(Dataset):
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
            *_
        ) = torch.load(self.data[idx])
        *inputs, is_transpose = self.tokenizer(inputs)
        return (
            mel.transpose(-1, -2),
            *inputs,
            is_transpose
        )


def collate_fn(batch):
    (
        mel,
        *inputs,
        is_transpose
    ) = tuple(zip(*batch))

    x_length = torch.LongTensor([len(x) for x in inputs[0]])
    inp_list = list()
    for i, inp in enumerate(inputs):
        x = pad_sequence(inp, batch_first=True)
        if is_transpose[0][i]:
            x = x.transpose(-1, -2)
        inp_list.append(x)

    y_length = torch.LongTensor([x.size(0) for x in mel])
    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)

    return (
        *inp_list,
        x_length,
        mel,
        y_length
    )