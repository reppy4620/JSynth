from ..conformer import ConformerModule


class PLModule:
    _d = {
        'conformer': ConformerModule
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.name](config)
