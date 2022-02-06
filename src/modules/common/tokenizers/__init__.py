from .ttslearn import TTSLearnTokenizer
from .paf import PAFTokenizer


class Tokenizer:
    _d = {
        'ttslearn': TTSLearnTokenizer,
        'paf': PAFTokenizer
    }

    @classmethod
    def from_config(cls, config):
        return cls._d[config.mode](**config)
