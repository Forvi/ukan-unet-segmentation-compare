from src.interfaces.metric import Metric


class IoUMetric(Metric):
    def __init__(self):
        pass


    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

