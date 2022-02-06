import torch
from nnmnkwii.io import hts
from ttslearn.tacotron.frontend.openjtalk import pp_symbols
from ttslearn.tacotron.frontend.openjtalk import text_to_sequence, phonemes, extra_symbols, num_vocab
from .base import TokenizerBase


class PPAddTokenizer(TokenizerBase):
    def __init__(self):
        self.extra_symbol_set = set(extra_symbols)

    def tokenize(self, text):
        inp = text_to_sequence(text.split())
        is_extra = [s in self.extra_symbol_set for s in text.split()]
        return torch.LongTensor(inp), torch.FloatTensor(is_extra)

    def __len__(self):
        return num_vocab()

    def extract(self, label_path, sr, y_length):
        label = hts.load(label_path)
        pp = pp_symbols(label.contexts)
        phoneme, prosody = list(), list()
        for p in pp:
            if p in phonemes:
                phoneme.append(p)
            elif p in extra_symbols:
                if p == '_':
                    phoneme.append('pau')
                else:
                    prosody.append(p)
            else:
                raise ValueError('p is invalid value')

        duration = self.extract_duration(label, sr, y_length)
        return (phoneme, prosody), duration
