import torch
from ttslearn.tacotron.frontend.openjtalk import text_to_sequence, extra_symbols, num_vocab
from .base import TokenizerBase


class TTSLearnTokenizer(TokenizerBase):
    def __init__(self):
        self.extra_symbol_set = set(extra_symbols)

    def tokenize(self, text):
        inp = text_to_sequence(text.split())
        is_extra = [s in self.extra_symbol_set for s in text.split()]
        print(inp)
        return torch.LongTensor(inp), torch.LongTensor(is_extra)

    def __len__(self):
        return num_vocab()
