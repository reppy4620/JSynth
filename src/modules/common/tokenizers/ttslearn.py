import torch
from nnmnkwii.io import hts
from ttslearn.tacotron.frontend.openjtalk import pp_symbols
from ttslearn.tacotron.frontend.openjtalk import text_to_sequence, extra_symbols, num_vocab
from .base import TokenizerBase


class TTSLearnTokenizer(TokenizerBase):
    def __init__(self):
        self.extra_symbol_set = set(extra_symbols)

    def tokenize(self, text):
        inp = text_to_sequence(text)
        is_extra = [s in self.extra_symbol_set for s in text]
        is_transpose = [0, 1]
        return torch.LongTensor(inp), torch.FloatTensor(is_extra).transpose(-1, -2), is_transpose

    def __len__(self):
        return num_vocab()

    def extract(self, label_path, sr, y_length):
        label = hts.load(label_path)
        phoneme = pp_symbols(label.contexts)

        duration = self.extract_duration(label, sr, y_length)
        final_duration = list()
        i = 0
        for p in phoneme:
            if p != '_' and p in extra_symbols:
                final_duration.append(0)
            else:
                final_duration.append(duration[i])
                i += 1
        assert len(phoneme) == len(final_duration)
        return phoneme, final_duration
