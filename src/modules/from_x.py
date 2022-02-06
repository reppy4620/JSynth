from omegaconf import OmegaConf

from .conformer import ConformerModule, validate as cfm_val
from .common.preprocessors import (
    NormalPreProcessor
)
from .common.tokenizers import (
    TTSLearnTokenizer,
    PAFTokenizer
)

modules = {
    'conformer': ConformerModule
}

val_fn = {
    'conformer': cfm_val
}

preprocessors = {
    'normal': NormalPreProcessor
}

tokenizers = {
    'ttslearn': TTSLearnTokenizer,
    'paf': PAFTokenizer
}


def module_from_config(config):
    module = modules[config.name](config)
    return module


def validate_from_args(args):
    config = OmegaConf.load(args.config)
    validate = val_fn[config.name]
    validate(args, config)


def preprocessor_from_config(config):
    preprocessor = preprocessors[config.preprocessor](config.preprocess)
    return preprocessor


def tokenizer_from_config(config):
    tokenizer = tokenizers[config.tokenizer]()
    return tokenizer
