from pytorch_lightning.callbacks import Callback


class IntervalCheckpoint(Callback):
    def __init__(self, interval):
        self.interval = interval

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.interval == 0:
            trainer.save_checkpoint(f'epoch_{trainer.current_epoch}.ckpt')
