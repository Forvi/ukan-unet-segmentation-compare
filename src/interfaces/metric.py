from abc import ABC, abstractmethod


class Metric(ABC):
    
    def __call__(self, *args, **kwds):
        pass