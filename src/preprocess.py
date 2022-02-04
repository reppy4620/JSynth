from argparse import ArgumentParser
from omegaconf import OmegaConf
from modules.from_x import preprocessor_from_config


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    preprocessor = preprocessor_from_config(config)
    preprocessor.run()


if __name__ == '__main__':
    main()
