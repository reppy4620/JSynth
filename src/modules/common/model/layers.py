import math
import torch
import torch.nn as nn


class TTSLearnEmbeddingLayer(nn.Module):
    def __init__(self, n_phoneme, channels):
        super(TTSLearnEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.emb = nn.Embedding(n_phoneme, channels)
        nn.init.normal_(self.emb.weight, 0.0, channels ** -0.5)

    def forward(self, x):
        x = self.emb(x) * self.scale
        x = x.transpose(-1, -2)
        return x


class PAFEmbeddingLayer(nn.Module):
    # def __init__(self, n_p, n_a, n_f, channels):
    #     super(PAFEmbeddingLayer, self).__init__()
    #     self.scale = math.sqrt(channels)
    #
    #     self.p_emb = nn.Embedding(n_p, channels)
    #     nn.init.normal_(self.p_emb.weight, 0.0, channels ** -0.5)
    #
    #     self.a_emb = nn.Embedding(n_a, channels)
    #     nn.init.normal_(self.a_emb.weight, 0.0, channels ** -0.5)
    #
    #     self.f_emb = nn.Embedding(n_f, channels)
    #     nn.init.normal_(self.f_emb.weight, 0.0, channels ** -0.5)
    #
    # def forward(self, p, a, f):
    #     p = self.p_emb(p) * self.scale
    #     a = self.a_emb(a) * self.scale
    #     f = self.f_emb(f) * self.scale
    #     x = torch.cat([p, a, f], dim=-1).transpose(-1, -2)
    #     return x
    def __init__(self, n_p, n_f, channels, **kwargs):
        super(PAFEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.phoneme_emb = nn.Embedding(n_p, channels)
        self.f2_emb = nn.Embedding(n_f, channels)
        nn.init.normal_(self.phoneme_emb.weight, 0.0, channels ** -0.5)
        nn.init.normal_(self.f2_emb.weight, 0.0, channels ** -0.5)

    def forward(self, phoneme, a1, f2):
        phoneme_emb = self.phoneme_emb(phoneme) * self.scale
        f2_emb = self.f2_emb(f2) * self.scale
        a1_emb = a1.unsqueeze(-1).expand(-1, -1, phoneme_emb.size(-1))
        x = torch.cat([phoneme_emb, f2_emb, a1_emb], dim=-1).transpose(-1, -2)
        return x


class PPAddEmbeddingLayer(nn.Module):
    def __init__(self, n_phoneme, n_prosody, channels):
        super(PPAddEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.phoneme_emb = nn.Embedding(n_phoneme, channels)
        nn.init.normal_(self.phoneme_emb.weight, 0.0, channels ** -0.5)

        self.prosody_emb = nn.Embedding(n_prosody, channels)
        nn.init.normal_(self.prosody_emb.weight, 0.0, channels ** -0.5)

    def forward(self, phoneme, prosody):
        phoneme = self.phoneme_emb(phoneme) * self.scale
        prosody = self.prosody_emb(prosody) * self.scale
        x = (phoneme + prosody).transpose(-1, -2)
        return x


class EmbeddingLayer(nn.Module):
    _d = {
        'ttslearn': TTSLearnEmbeddingLayer,
        'paf': PAFEmbeddingLayer,
        'pp_add': PPAddEmbeddingLayer
    }

    def __init__(self, mode, **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.emb = self._d[mode](**kwargs)

    def forward(self, *args, **kwargs):
        return self.emb(*args, **kwargs)