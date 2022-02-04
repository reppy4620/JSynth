from abc import abstractmethod


class TokenizerBase:

    def __call__(self, text):
        self.tokenize(text)

    @abstractmethod
    def tokenize(self, text):
        pass
