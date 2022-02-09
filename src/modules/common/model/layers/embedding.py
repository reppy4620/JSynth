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
    def __init__(self, n_p, n_a, n_f, channels):
        super(PAFEmbeddingLayer, self).__init__()
        self.scale = math.sqrt(channels)

        self.p_emb = nn.Embedding(n_p, channels)
        nn.init.normal_(self.p_emb.weight, 0.0, channels ** -0.5)

        self.a_emb = nn.Embedding(n_a, channels)
        nn.init.normal_(self.a_emb.weight, 0.0, channels ** -0.5)

        self.f_emb = nn.Embedding(n_f, channels)
        nn.init.normal_(self.f_emb.weight, 0.0, channels ** -0.5)

    def forward(self, p, a, f):
        p = self.p_emb(p) * self.scale
        a = self.a_emb(a) * self.scale
        f = self.f_emb(f) * self.scale
        x = torch.cat([p, a, f], dim=-1).transpose(-1, -2)
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


class RelPositionalEncoding(nn.Module):
    def __init__(self, channels, dropout=0.1, max_len=10000):
        super(RelPositionalEncoding, self).__init__()
        self.d_model = channels
        self.scale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(2) >= x.size(2) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.transpose(-1, -2).to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            :,
            self.pe.size(2) // 2 - x.size(2) + 1 : self.pe.size(2) // 2 + x.size(2),
        ]
        return x, self.dropout(pos_emb)
