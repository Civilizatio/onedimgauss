import time
import numpy as np


class Timer:
    """Record time the code cost."""

    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._elapsed_time = None

    def start(self):
        self._start_time = time.time()

    def stop(self):
        self._end_time = time.time()

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._end_time = time.time()
        self._elapsed_time = self._end_time - self._start_time
        return False

    @property
    def elapsed_time(self):
        if self._start_time is not None and self._end_time is not None:
            self._elapsed_time = self._end_time - self._start_time
        else:
            raise ValueError("Timer wrong!")
        return self._elapsed_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n
        self.len = [0] * n 

    def add(self, tensor):
        if len(tensor) != len(self.data):
            raise ValueError("Dimension of input tensor does not match accumulator dimension")
        for i, value in enumerate(tensor):
            if not np.isnan(value):
                self.data[i] += value
                self.len[i] += 1

    def reset(self):
        self.data = [0.0] * len(self.data)
        self.len = [0] * len(self.len)

    def average(self):
        return [data / count if count > 0 else float('nan') for data, count in zip(self.data, self.len)]

    def __getitem__(self, idx):
        return self.data[idx]