from pathlib import Path
from argparse import ArgumentParser
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from modules import PLModule
from callbacks import IntervalCheckpoint


def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, f'{config.output_dir}/config.yaml')

    module = PLModule.from_config(config)

    best_cp = ModelCheckpoint(
        dirpath=config.output_dir,
        filename='best',
        monitor=config.train.monitor,
        save_last=True,
        verbose=True
    )
    interval_cp = IntervalCheckpoint(config.train.save_interval)

    csv_logger = CSVLogger(
        save_dir=f'{config.output_dir}',
        name=f'csv_logs',
        flush_logs_every_n_steps=100000,
        version='1'
    )
    tb_logger = TensorBoardLogger(
        save_dir=f'{config.output_dir}',
        name=f'tb_logs',
        version='1'
    )

    last_ckpt_path = output_dir / 'last.ckpt'
    trainer = Trainer(
        logger=[csv_logger, tb_logger],
        callbacks=[best_cp, interval_cp],
        gpus=1,
        max_epochs=config.train.num_epochs,
        log_every_n_steps=10,
        gradient_clip_val=5.0,
        resume_from_checkpoint=last_ckpt_path if last_ckpt_path.exists() else None
    )
    module.trainer = trainer
    trainer.fit(module)


if __name__ == '__main__':
    main()
