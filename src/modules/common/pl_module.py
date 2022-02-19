from ..conformer import ConformerModule
from ..glow_tts import GlowTTSModule
from ..glow_tts_f0 import GlowTTSWithF0Module
from ..grad_tts import GradTTSModule
from ..grad_tts_f0 import GradTTSWithF0Module


class PLModule:
    _d = {
        'conformer': ConformerModule,
        'glow_tts': GlowTTSModule,
        'glow_tts_f0': GlowTTSWithF0Module,
        'grad_tts': GradTTSModule,
        'grad_tts_f0': GradTTSWithF0Module,
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.name](config)
