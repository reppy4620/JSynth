from .all import AllDataset, collate_fn as all_collate


class TTSDataset:
    _d = {
        'all': (AllDataset, all_collate)
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.mode]
