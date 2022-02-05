from abc import abstractmethod


class TokenizerBase:

    def __call__(self, text):
        return self.tokenize(text)

    @abstractmethod
    def tokenize(self, text):
        pass
