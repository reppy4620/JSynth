from .normal import NormalDataset, collate_fn as normal_collate


class TTSDataset:
    _d = {
        'normal': (NormalDataset, normal_collate)
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.mode]
