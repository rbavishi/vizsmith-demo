import itertools
from typing import Dict, Optional, Iterator, Callable, Any, List

import attr
import matplotlib.pyplot as plt

from synthesis.base_instantiator import BaseInstantiator
from synthesis.query import Query
from synthesis.utils import deepcopy_args
from utilities.matplotlib_utils import run_viz_code_matplotlib_mp, serialize_fig, run_viz_code_matplotlib


@attr.s(cmp=False, repr=False)
class SimpleInstantiator(BaseInstantiator):
    @staticmethod
    def instantiate(query: Query,
                    viz_functions: List[Dict],
                    timeout: Optional[int] = None,
                    serializer: Callable[[plt.Figure], Any] = None) -> Iterator[Dict]:

        #  Simply go in increasing order of length (Occam's razor)
        for viz_function in sorted(viz_functions, key=lambda t: len(t['code'])):
            df_args = viz_function['df_args']
            col_args = viz_function['col_args']
            if len(col_args) != len(query.requested_cols):
                continue

            for df_asgn in itertools.permutations(list(query.provided_dfs)):
                for col_asgn in itertools.permutations(query.requested_cols):
                    df_idxes = list(range(len(df_asgn)))
                    df_m = dict(zip(df_args, df_asgn))
                    col_m = dict(zip(col_args, col_asgn))

                    #  Get the variable name mapping for the df arguments
                    df_args_mapping = dict(zip(df_args, df_idxes))
                    col_arg_mapping = col_m.copy()

                    try:
                        #  We opt to return the png directly as pickling figure objects
                        #  can be tricky with different ipykernel backends.
                        if serializer is not None:
                            result = run_viz_code_matplotlib_mp(viz_function['code'],
                                                                deepcopy_args({**df_m, **col_m}),
                                                                serializer=serializer)

                            if result is not None:
                                yield {
                                    'serialized': result,
                                    'code': viz_function['code'],
                                    'df_args_mapping': df_args_mapping,
                                    'col_args_mapping': col_arg_mapping,
                                }
                        else:
                            fig = run_viz_code_matplotlib_mp(viz_function['code'],
                                                             deepcopy_args({**df_m, **col_m}))
                            if fig is not None:
                                yield {
                                    'fig': fig,
                                    'code': viz_function['code'],
                                    'df_args_mapping': df_args_mapping,
                                    'col_args_mapping': col_arg_mapping,
                                }

                    except Exception as e:
                        pass
