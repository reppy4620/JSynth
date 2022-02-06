import re

import torch
from nnmnkwii.io import hts

from .base import TokenizerBase
from .phonemes import phonemes


class PAFTokenizer(TokenizerBase):

    def __init__(self, state_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_size = state_size

        self.p_dict = self.load_dictionary()
        self.a_dict = self.build_num_dict(start=-15, end=9)
        self.f_dict = self.build_num_dict(start=0, end=16)

    @staticmethod
    def load_dictionary():
        dictionary = dict()
        for i, w in enumerate(phonemes):
            dictionary[w] = i
        return dictionary

    @staticmethod
    def build_num_dict(start, end):
        d = {str(k): i for i, k in enumerate(range(start, end + 1), start=1)}
        d['xx'] = len(d) + 1
        return d

    def tokenize(self, p, a, f):
        p_id = [self.p_dict[s] for s in p]
        a_id = [self.a_dict[s] for s in a]
        f_id = [self.f_dict[s] for s in f]

        p_id = torch.LongTensor(p_id)
        a_id = torch.LongTensor(a_id)
        f_id = torch.LongTensor(f_id)
        return p_id, a_id, f_id

    def __len__(self):
        return len(self.p_dict)

    def extract(self, label_path, sr, y_length):
        label = hts.load(label_path)

        p_list = list()
        a_list = list()
        f_list = list()
        for context in label.contexts:
            if context.split("-")[1].split("+")[0] == "pau":
                p_list += ["pau"]
                continue
            elif context.split("-")[1].split("+")[0] == "sil":
                continue
            paf = re.findall(r"\-(.*?)\+.*?\/A:([0-9\-]+).*?\/F:.*?_([0-9]+)", context)
            if len(paf) == 1:
                p_list += [paf[0][0]]
                a_list += [paf[0][1]]
                f_list += [paf[0][2]]
        duration = self.extract_duration(label, sr, y_length)
        return (p_list, a_list, f_list), duration
