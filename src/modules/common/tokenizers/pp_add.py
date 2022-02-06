import torch
from nnmnkwii.io import hts
from ttslearn.tacotron.frontend.openjtalk import pp_symbols
from ttslearn.tacotron.frontend.openjtalk import phonemes, extra_symbols, num_vocab
from .base import TokenizerBase


class PPAddTokenizer(TokenizerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phoneme_dict = {s: i for i, s in enumerate(['<pad>'] + phonemes)}
        self.prosody_dict = {s: i for i, s in enumerate(['<pad>'] + extra_symbols)}

    def tokenize(self, inputs):
        phoneme, prosody = inputs
        phoneme = [self.phoneme_dict[s] for s in phoneme]
        prosody = [self.prosody_dict[s] for s in prosody]
        is_transpose = [0, 0]
        return torch.LongTensor(phoneme), torch.FloatTensor(prosody), is_transpose

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
