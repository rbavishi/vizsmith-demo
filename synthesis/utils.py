from typing import Dict, Any

import pandas as pd


def deepcopy_args(args: Dict[str, Any]) -> Dict[str, Any]:
    res = {}
    for k, v in args.items():
        if isinstance(v, pd.DataFrame):
            res[k] = v.copy()
        elif isinstance(v, (str, int, float)):
            res[k] = v
        else:
            raise NotImplementedError(f"Cannot copy type {type(v)}")

    return res
