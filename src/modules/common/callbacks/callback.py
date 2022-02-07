from abc import abstractmethod


class Callback:
    @abstractmethod
    def on_epoch_start(self, trainer):
        pass

    @abstractmethod
    def on_epoch_end(self, trainer):
        pass
