from abc import ABC, abstractmethod


class AbstractReader(ABC):
    @abstractmethod
    def read(self):
        pass
