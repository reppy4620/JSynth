from abc import abstractmethod


class TransformBase:

    def __init__(self, params=None):
        self.params = params

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass
