from abc import ABC, abstractmethod
from typing import List, Dict

from synthesis.query import Query


class BaseSearcher(ABC):
    @abstractmethod
    def search(self, query: Query) -> List[Dict]:
        pass
