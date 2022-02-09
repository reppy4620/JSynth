from ..conformer import ConformerModule
from ..glow_tts import GlowTTSModule
from ..grad_tts import GradTTSModule


class PLModule:
    _d = {
        'conformer': ConformerModule,
        'glow_tts': GlowTTSModule,
        'grad_tts': GradTTSModule
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.name](config)
