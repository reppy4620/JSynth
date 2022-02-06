from .noam import NoamLR


class Scheduler:
    _d = {
        'noam': NoamLR
    }

    @classmethod
    def from_config(cls, config, last_epoch):
        return cls._d[config.mode](**config, last_epoch=last_epoch)
