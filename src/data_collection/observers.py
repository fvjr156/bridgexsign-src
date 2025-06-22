from abc import ABC, abstractmethod
from typing import Any, List

class AbstractObserver(ABC):
    @abstractmethod
    def update(self, event_type: str, data: Any):
        pass

class Subject:
    def __init__(self):
        self._observers: List[AbstractObserver] = []

    def attach(self, observer: AbstractObserver):
        self._observers.append(observer)

    def detach(self, observer: AbstractObserver):
        self._observers.remove(observer)

    def notifyAllObservers(self, event_type: str, data: Any = None):
        for observer in self._observers:
            observer.update(event_type, data)