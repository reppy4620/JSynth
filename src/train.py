from argparse import ArgumentParser
from omegaconf import OmegaConf
from pathlib import Path
from modules.common.trainer import Trainer


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / 'last.ckpt'
    trainer = Trainer(
        config,
        ckpt_path if ckpt_path.exists() else None
    )
    trainer.run()


if __name__ == '__main__':
    main()
