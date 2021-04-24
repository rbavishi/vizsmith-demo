"""
Multprocessing helpers and add-ons
"""
import collections
import hashlib
import multiprocessing
import os
import pickle
from typing import Callable, Dict, Any, Hashable

import tqdm


def fault_tolerant_imap_unordered(func: Callable,
                                  task_dict: Dict[Hashable, Any],
                                  key: str,
                                  num_processes: int = 1,
                                  maxtasksperchild=None,
                                  method: str = 'spawn'):
    """
    A fault-tolerant version of imap_unordered which is able to resume from where it last stopped due to an exception.
    The `task_dict` argument must be a dictionary with keys as ID strings and values as the arguments to func.
    The `key` argument must be a string identifying the overall task.
    The `method` arg should be one of 'spawn' and 'fork'.

    The *only* argument to `func` will be a tuple whose first element is the task ID and the second argument is
    the value of the corresponding entry in `task_dict`.
    The `func` callable *must* return a dictionary with an entry for 'id' corresponding to the 'id' of the task, and
    an entry 'value' corresponding to actual result of `func`.
    It may optionally contain an entry for 'status' that will be displayed in the progress bar.

    Note that you must ensure the result of each sub-task is picklable and disk usage is not an issue
    when storing each of the results individually in a pickle file.
    :param func:
    :param task_dict:
    :param key:
    :param num_processes:
    :param maxtasksperchild:
    :param method:
    :return:
    """
    id_hash = hashlib.sha256(pickle.dumps(sorted(task_dict.keys()))).hexdigest()
    cache_path = f"/tmp/fault_tolerant_imap_unordered_{key}_{id_hash}.pkl"

    results = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                while True:
                    entry = pickle.load(f)
                    results.update(entry)
        except:
            pass

    fp_cache = open(cache_path, 'wb')
    for k, v in results.items():
        pickle.dump({k: v}, file=fp_cache)

    fp_cache.flush()

    to_process = {k: v for k, v in task_dict.items() if k not in results}
    status_dict = collections.Counter()
    with tqdm.tqdm(total=len(to_process), dynamic_ncols=True) as pbar:
        with multiprocessing.get_context(method).Pool(num_processes,
                                                      maxtasksperchild=maxtasksperchild) as pool:
            for res in pool.imap_unordered(func, to_process.items()):
                task_id = res['id']
                value = res['value']
                results[task_id] = value

                pickle.dump({task_id: value}, file=fp_cache)
                fp_cache.flush()

                if 'status' in res:
                    status_dict[res['status']] += 1
                    pbar.set_postfix(**status_dict)

                pbar.update(1)

    os.remove(cache_path)
    return results
