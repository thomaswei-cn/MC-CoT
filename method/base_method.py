from utils.register import register_class
from abc import abstractmethod

@register_class("BaseMethod")
class BaseMethod:
    def __init__(self):
        pass

    @abstractmethod
    def run(self):
        pass