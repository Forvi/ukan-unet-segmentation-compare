from abc import ABC, abstractmethod


class Metric(ABC):
    
    @abstractmethod
    def update():
        pass


    @abstractmethod
    def reset():
        pass


    def compute():
        pass