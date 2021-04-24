import io
from concurrent.futures import TimeoutError
from typing import Dict, Any, Optional, Callable

from matplotlib import pyplot as plt
from pebble import concurrent


def serialize_fig(fig: plt.Figure, format: str = 'png', tight: bool = True):
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, bbox_inches='tight', format='png')
    else:
        fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.read()


def run_viz_code_matplotlib(code: str, args: Dict[str, Any],
                            func_name: str = 'visualization',
                            other_globals: Optional[Dict] = None,
                            disable_seaborn_randomization: bool = True,
                            serializer: Callable[[plt.Figure], Any] = None):
    """
    Helper to execute matplotlib-based viz code and return the figure.
    :param code:
    :param args:
    :param func_name:
    :param other_globals:
    :param disable_seaborn_randomization:
    :param serializer:
    :return:
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        plt.style.use('default')
    except:
        pass
    try:
        mpl.rcParams.update(mpl.rcParamsDefault)
    except:
        pass

    if disable_seaborn_randomization:
        try:
            orig_random_seed_fn = sns.algorithms._handle_random_seed

            def wrapper(seed):
                return orig_random_seed_fn(0)

            sns.algorithms._handle_random_seed = wrapper
        except:
            pass

    try:
        orig_fig = plt.figure()
        m = {}
        if other_globals is not None:
            m.update(other_globals)

        exec(code, m, m)
        func = m[func_name]
        func(**args)
        fig = plt.gcf()
        if fig is not orig_fig:
            plt.close(orig_fig)

        if serializer is not None:
            result = serializer(fig)
            plt.close(fig)
            return result

        return fig

    finally:
        try:
            plt.style.use('default')
        except:
            pass
        try:
            mpl.rcParams.update(mpl.rcParamsDefault)
        except:
            pass

        if disable_seaborn_randomization:
            try:
                sns.algorithms._handle_random_seed = orig_random_seed_fn
            except:
                pass


def run_viz_code_matplotlib_mp(code: str, args: Dict[str, Any],
                               func_name: str = 'visualization',
                               other_globals: Optional[Dict] = None,
                               disable_seaborn_randomization: bool = True,
                               serializer: Callable[[plt.Figure], Any] = None,
                               timeout: Optional[int] = None):

    func = concurrent.process(timeout=timeout)(run_viz_code_matplotlib)
    future = func(code=code, args=args, func_name=func_name, other_globals=other_globals,
                  disable_seaborn_randomization=disable_seaborn_randomization, serializer=serializer)

    try:
        result = future.result()
        return result

    except TimeoutError:
        return None


def turn_off_multiple_open_figure_warning():
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParamsDefault.update({'figure.max_open_warning': 0})
