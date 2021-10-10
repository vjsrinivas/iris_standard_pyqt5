import sys
import abc # Abstract Base Classes

class Model(abc.ABC):
    def __init__(self, *network_config) -> None:
        self.initialize(*network_config)
    
    @abc.abstractclassmethod
    def run(self, input):
        raise NotImplementedError

    @abc.abstractclassmethod
    def initialize(self, *kwargs):
        raise NotImplementedError

    @abc.abstractclassmethod
    def deinitialize(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def draw(self, pred):
        raise NotImplementedError

    def __call__(self):
        pred = self.run()
        return pred

