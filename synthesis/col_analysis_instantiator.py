import collections
import itertools
import time
from typing import List, Dict, Optional, Callable, Any, Iterator, Tuple

import attr
from matplotlib import pyplot as plt

from synthesis.base_instantiator import BaseInstantiator
from synthesis.query import Query
from synthesis.utils import deepcopy_args
from utilities.matplotlib_utils import run_viz_code_matplotlib_mp


def _get_valid_assignments(candidates_map: List[Tuple[Any, List[Tuple[float, str]]]],
                           _depth: int = 0,
                           _cur_map: Optional[Dict] = None):
    if _cur_map is None:
        _cur_map = {'score': 0}

    if _depth == len(candidates_map):
        m = _cur_map.copy()
        yield m.pop('score') / max(1, len(m)), m
        return

    q_col = candidates_map[_depth][0]
    for cand_score, cand in candidates_map[_depth][1]:
        if cand not in _cur_map:
            new_map = _cur_map.copy()
            new_map[cand] = q_col
            new_map['score'] += cand_score
            yield from _get_valid_assignments(candidates_map, _depth=_depth + 1, _cur_map=new_map)


def get_col_compatibility_score(df_col: Any, df_metadata: Dict, orig_metadata: Dict):
    hl_dtype = df_metadata['high_level_data_types'][df_col]
    ll_dtype = df_metadata['low_level_data_types'][df_col]
    has_null = df_metadata['has_null'][df_col]

    o_hl_dtype = orig_metadata['high_level_dtype']
    o_ll_dtype = orig_metadata['low_level_dtype']
    o_has_null = orig_metadata['has_null']

    score = 0
    hl_dtype = set(hl_dtype.split("/")) if "/" in hl_dtype else {hl_dtype}
    o_hl_dtype = set(o_hl_dtype.split("/")) if "/" in o_hl_dtype else {o_hl_dtype}

    if len(hl_dtype & o_hl_dtype) > 0:
        score += 1.0
    else:
        return 0.0

    if len(ll_dtype) > 0 or len(o_ll_dtype) > 0:
        score += len(ll_dtype & o_ll_dtype) / (len(ll_dtype | o_ll_dtype))

    if has_null == o_has_null:
        score += 0.1

    return score


def get_possible_column_assignments(query: Query, viz_function: Dict, df_index_to_arg: Dict[int, str]):
    df_arg_to_index = {v: k for k, v in df_index_to_arg.items()}
    implicit_cols = viz_function['col_analysis']['implicit_cols']
    col_args_metadata = viz_function['col_analysis']['metadata_col_args']

    req_cols_to_dfs = collections.defaultdict(list)
    for df_index, cols in enumerate(query.requested_cols):
        for col in cols:
            req_cols_to_dfs[col].append(df_index_to_arg[df_index])

    if len(implicit_cols) == 0:
        #  Need an exact correspondence between the args and the requested columns
        if len(req_cols_to_dfs) != len(viz_function['col_args']):
            return

        if len(viz_function['col_args']) == 0:
            yield 100.0, {}, False
            return

        candidates_map = collections.defaultdict(list)
        for q_col, df_keys in req_cols_to_dfs.items():
            for t_col in viz_function['col_args']:
                if any((df_k, t_col) not in col_args_metadata for df_k in df_keys):
                    continue

                score = 0
                for df_k in df_keys:
                    orig_metadata = col_args_metadata[(df_k, t_col)]
                    df_metadata = query.get_df_metadata(df_arg_to_index[df_k])
                    score += get_col_compatibility_score(q_col, df_metadata, orig_metadata)

                score /= len(df_keys)
                if score > 0:
                    candidates_map[q_col].append((score, t_col))

            if len(candidates_map[q_col]) == 0:
                return

        #  Return all possible assignments in descending order of score, with ties broken in random order.
        asgns = sorted(_get_valid_assignments(list(candidates_map.items())), key=lambda x: -x[0])
        for score, asgn in asgns:
            yield score, asgn, False
    else:
        if len(viz_function['col_args']) > len(set(sum(query.requested_cols, []))):
            return
        #  Get the list of columns that are eligible for implicit visualization
        conditions = collections.defaultdict(set)

        for (df_key, hl_dtype, ll_dtype, _), entries in implicit_cols.items():
            conditions[df_key].add((hl_dtype, ll_dtype))

        eligible_implicit_cols = set()

        for df_key, condition in conditions.items():
            df_index = df_arg_to_index[df_key]
            df_metadata = query.get_df_metadata(df_index)
            for col in query.requested_cols[df_index]:
                hl_dtype = df_metadata['high_level_data_types'][col]
                ll_dtype = df_metadata['low_level_data_types'][col]
                if any((hl_bit, ll_dtype) in condition for hl_bit in hl_dtype.split('/')):
                    eligible_implicit_cols.add((df_key, col))

        forced_args = set()
        for df_idx, req_cols in enumerate(query.requested_cols):
            for req_col in req_cols:
                if (df_index_to_arg[df_idx], req_col) not in eligible_implicit_cols:
                    forced_args.add(req_col)

        #  Decide what to put in the col args
        candidates_map = collections.defaultdict(list)
        for t_col in viz_function['col_args']:
            for q_col, df_keys in req_cols_to_dfs.items():
                if any((df_k, t_col) not in col_args_metadata for df_k in df_keys):
                    continue

                score = 0
                for df_k in df_keys:
                    orig_metadata = col_args_metadata[(df_k, t_col)]
                    df_metadata = query.get_df_metadata(df_arg_to_index[df_k])
                    score += get_col_compatibility_score(q_col, df_metadata, orig_metadata)

                score /= len(df_keys)
                if score > 0:
                    candidates_map[t_col].append((score, q_col))

            if len(candidates_map[t_col]) == 0:
                return

        #  Return all possible assignments in descending order of score, with ties broken in random order.
        asgns = sorted(_get_valid_assignments(list(candidates_map.items())), key=lambda x: -x[0])
        for score, asgn in asgns:
            asgn = {v: k for k, v in asgn.items()}
            values = set(asgn.values())
            if not forced_args.issubset(values):
                continue

            yield score, asgn, True
        #
        # print(eligible_implicit_cols)
        # print(candidates_map)
        # print(viz_function['code'])
        # print(viz_function['key'])


@attr.s(cmp=False, repr=False)
class ColAnalysisInstantiator(BaseInstantiator):
    """
    Uses column-level analysis to better instantiate viz_functions
    """

    @staticmethod
    def instantiate(query: Query,
                    viz_functions: List[Dict],
                    timeout: Optional[int] = None,
                    per_run_timeout: Optional[int] = None,
                    serializer: Callable[[plt.Figure], Any] = None,
                    stats: Optional[Dict] = None) -> Iterator[Dict]:

        if stats is None:
            stats = {}

        stats['num_failures'] = 0
        stats['num_successes'] = 0

        start_time = time.time()

        #  Get the best scoring df and col assignments, with ties broken by length of code in viz_function.
        candidates = []
        for viz_function in viz_functions:
            for df_asgn in itertools.permutations(list(viz_function['df_args'].keys())):
                df_index_to_arg = dict(zip(list(range(len(query.provided_dfs))), df_asgn))
                df_arg_to_index = {v: k for k, v in df_index_to_arg.items()}
                for score, col_asgn, take_subset in get_possible_column_assignments(query, viz_function, df_index_to_arg):
                    if take_subset:
                        provided_dfs = [df[query.requested_cols[idx]] for idx, df in enumerate(query.provided_dfs)]
                    else:
                        provided_dfs = query.provided_dfs

                    args = {
                        **{v: provided_dfs[k] for k, v in df_index_to_arg.items()},
                        **col_asgn,
                    }
                    candidates.append(([-score, len(viz_function['code'])], viz_function, args, df_arg_to_index, col_asgn))

        stats['num_candidates'] = len(candidates)

        for _, viz_function, args, df_args_mapping, col_args_mapping in sorted(candidates, key=lambda x: x[0]):
            if timeout is not None and time.time() - start_time > timeout:
                break

            try:
                #  We opt to return the png directly as pickling figure objects
                #  can be tricky with different ipykernel backends.
                if serializer is not None:
                    run_start = time.time()
                    result = run_viz_code_matplotlib_mp(viz_function['code'],
                                                        deepcopy_args(args),
                                                        serializer=serializer,
                                                        timeout=per_run_timeout)
                    run_end = time.time()

                    if result is not None:
                        stats['num_successes'] += 1
                        yield {
                            'serialized': result,
                            'key': viz_function['key'],
                            'code': viz_function['code'],
                            'df_args_mapping': df_args_mapping,
                            'col_args_mapping': col_args_mapping,
                            'run_time': run_end - run_start,
                        }
                    else:
                        stats['num_failures'] += 1
                else:
                    run_start = time.time()
                    fig = run_viz_code_matplotlib_mp(viz_function['code'],
                                                     deepcopy_args(args),
                                                     timeout=per_run_timeout)
                    run_end = time.time()

                    if fig is not None:
                        stats['num_successes'] += 1
                        yield {
                            'fig': fig,
                            'key': viz_function['key'],
                            'code': viz_function['code'],
                            'df_args_mapping': df_args_mapping,
                            'col_args_mapping': col_args_mapping,
                            'run_time': run_end - run_start,
                        }
                    else:
                        stats['num_failures'] += 1

            except Exception as e:
                # import traceback
                # print(traceback.format_exc())
                stats['num_failures'] += 1


if __name__ == '__main__':
    import pickle
    import tqdm
    import pandas as pd
    from synthesis.simple_code_searcher import SimpleCodeSearcher
    from utilities.matplotlib_utils import serialize_fig

    with open('../filtered_viz_functions_indexable.pkl', 'rb') as f:
        ts = pickle.load(f)

    df = pd.DataFrame({
        'Category': ['X', 'Y', 'X', 'X', 'Y', 'X'],
        'Number': [1, 2, 34, 5, -1, -2],
        'Binary': [0, 1, 0, 1, 0, 0],
        'ID': [1,2,3,4,5,6],
    })

    searcher = SimpleCodeSearcher(ts)
    query = Query('lineplot', [df], [['Number']])
    stats = {}
    ans = []
    ist = ColAnalysisInstantiator()
    for t in tqdm.tqdm(list(searcher.search(query))[:10]):
        s = {}
        ans.extend(ist.instantiate(query, [t], per_run_timeout=10,
                                   stats=s))
        for k, v in s.items():
            if k not in stats:
                stats[k] = 0

            stats[k] += v

    print(len(ans))
    print(stats)

    for i in ans:
        print(i['code'])
        print(i['col_args_mapping'])
        print(i['run_time'])

