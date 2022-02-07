from collections import defaultdict


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def mean(self):
        return self.avg


class Tracker(defaultdict):
    def __init__(self):
        super().__init__(AverageMeter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k].update(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)
