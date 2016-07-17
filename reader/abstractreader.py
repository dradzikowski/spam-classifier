import os
from abc import ABC, abstractmethod

MODULE_DIR = os.path.abspath(os.path.join('.'))


class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        pass
