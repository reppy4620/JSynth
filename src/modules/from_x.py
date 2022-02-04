from omegaconf import OmegaConf

from .cfm import ConformerModule, validate as ae_val
from .common.preprocessors import NormalPreProcessor

modules = {
    'cfm': ConformerModule
}

val_fn = {
    'cfm': ae_val
}

preprocessors = {
    'cfm': NormalPreProcessor
}


def module_from_config(config):
    module = modules[config.name](config)
    return module


def validate_from_args(args):
    config = OmegaConf.load(args.config)
    validate = val_fn[config.name]
    validate(args, config)


def preprocessor_from_config(config):
    preprocessor = preprocessors[config.name](config.preprocess)
    return preprocessor
