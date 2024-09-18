from abc import abstractmethod
from utils.register import register_class


class BaseLanguageEngine:
    def __init__(self):
        pass

    @abstractmethod
    def get_response(self, user_input):
        pass
