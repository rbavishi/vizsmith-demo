import attr
import pandas as pd
from typing import List, Any, Optional

from utilities.df_utils import compute_df_metadata


@attr.s(cmp=False, repr=False)
class Query:
    query_str: str = attr.ib()
    provided_dfs: List[pd.DataFrame] = attr.ib()
    requested_cols: List[List] = attr.ib()

    _df_metadata = attr.ib(init=False, factory=dict)

    def get_df_metadata(self, index: int):
        if index not in self._df_metadata:
            self._df_metadata[index] = compute_df_metadata(self.provided_dfs[index])

        return self._df_metadata[index]

