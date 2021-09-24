import torch.nn as nn


from abc import abstractmethod

from typing import List, Callable, Union, Any, TypeVar, Tuple

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')  # Type 지정


class FlowFrame(nn.Module):
    def __init__(self):
        super().__init__()

    def g(self, z: Tensor):  # z -> x
        raise NotImplementedError

    def f(self, x: Tensor):  # x -> z, log_prob
        raise NotImplementedError

    def log_prob(self, x: Tensor):
        raise NotImplementedError

    def sample(self, num):
        raise NotImplementedError

    @abstractmethod  # 이 Method는 꼭 구현해야함을 강제.
    def forward(self, x):
        pass


class PriorFrame():
    def __init__(self):
        pass

    def log_prob(self, input):
        raise NotImplementedError

    def sample(self, size, dim):
        raise NotImplementedError