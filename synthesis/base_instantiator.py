from abc import abstractmethod, ABC

import attr
import matplotlib.pyplot as plt
from typing import Dict, Optional, Iterator, Callable, Any, List

from synthesis.query import Query


@attr.s(cmp=False, repr=False)
class BaseInstantiator(ABC):
    @staticmethod
    @abstractmethod
    def instantiate(query: Query,
                    viz_functions: List[Dict],
                    timeout: Optional[int] = None,
                    serializer: Callable[[plt.Figure], Any] = None) -> Iterator[Dict]:
        """
        Should be a generator which yields a dictionary with at least the following entries:
        'fig': The figure object if serializer is None,
        'serialized': The serialization of the figure if serializer is not None,
        'df_args_mapping': A dictionary from df args to integers corresponding to indices of query.provided_dfs
        'col_args_mapping': A dictionary from column args to actual column strings
        :param query:
        :param viz_functions:
        :param timeout:
        :param serializer:
        :return:
        """
