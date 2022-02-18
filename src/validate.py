from argparse import ArgumentParser
from omegaconf import OmegaConf

from modules.conformer import validate as cfm_validate
from modules.glow_tts import validate as glow_tts_validate
from modules.glow_tts_f0 import validate as glow_tts_f0_validate


validate_fn = {
    'conformer': cfm_validate,
    'glow_tts': glow_tts_validate,
    'glow_tts_f0': glow_tts_f0_validate
}


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--vocoder_path', type=str)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    validate_fn[config.name](args, config)


if __name__ == '__main__':
    main()
