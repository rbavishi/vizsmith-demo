import base64
import itertools
import multiprocessing
import os
import pickle
import sys
import threading
import time
from typing import List, Dict, Union, Callable

import attr
import ipywidgets as widgets
import pandas as pd
import matplotlib as mpl
from IPython.core.display import display, Javascript
from ipywidgets import Button, Layout
from pygments import highlight
from pygments import lexers
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name
from viz_synthesis_widget import VizSynthesisWidget

import common
from synthesis.base_instantiator import BaseInstantiator
from synthesis.base_searcher import BaseSearcher
from synthesis.nl_searcher import NaturalLanguageSearcher
from synthesis.query import Query
from synthesis.simple_code_searcher import SimpleCodeSearcher
from synthesis.simple_instantiator import SimpleInstantiator
from synthesis.simple_nl_plus_code_searcher import SimpleNLPlusCodeSearcher, WhooshNLPlusCodeSearcher
from utilities.matplotlib_utils import turn_off_multiple_open_figure_warning, serialize_fig

turn_off_multiple_open_figure_warning()
_searcher_cache = {}


def create_expanded_button(description, button_style, icon='',
                           height='auto', width='auto'):
    return Button(description=description,
                  button_style=button_style,
                  layout=Layout(height='auto', width='auto'),
                  icon=icon)


def read_image(path: str):
    with open(path, 'rb') as f:
        return f.read()


def get_html(code: str):
    lexer = lexers.get_lexer_by_name('python')
    style = get_style_by_name('default')
    html_formatter = HtmlFormatter(full=False, style=style, noclasses=True)
    return highlight(code, lexer, html_formatter)


def create_code_cell(code='', where='below'):
    """
    Sourced from : https://github.com/ipython/ipython/issues/4983
    Create a code cell in the IPython Notebook.

    Parameters
    code: unicode
        Code to fill the new code cell with.
    where: unicode
        Where to add the new code cell.
        Possible values include:
            at_bottom
            above
            below
    """
    encoded_code = (base64.b64encode(str.encode(code))).decode()
    display(Javascript("""
        var code = IPython.notebook.insert_cell_{0}('code');
        code.set_text(atob("{1}"));
    """.format(where, encoded_code)))


def check_rules(fig):
    """
    Check if fig meets criteria to be returned as a result. Figure must have been drawn already (using a savefig for ex.)
    :param fig:
    :return:
    """
    for ax in fig.axes:
        txts = {
            'xaxis_label': '',
            'yaxis_label': '',
            'xticks': [],
            'yticks': [],
            'legend': [],
        }

        for o in ax.findobj():
            if isinstance(o, mpl.axis.XAxis):
                try:
                    txts['xaxis_label'] = o.get_label()
                except:
                    pass

            if isinstance(o, mpl.axis.YAxis):
                try:
                    txts['yaxis_label'] = o.get_label()
                except:
                    pass

            if isinstance(o, mpl.axis.XTick):
                for t in o.findobj(mpl.text.Text):
                    if t.get_visible() and t.get_text():
                        txts['xticks'].append(t)

            if isinstance(o, mpl.axis.YTick):
                for t in o.findobj(mpl.text.Text):
                    if t.get_visible() and t.get_text():
                        txts['yticks'].append(t)

            if isinstance(o, mpl.legend.Legend):
                for t in o.findobj(mpl.text.Text):
                    if t.get_visible() and t.get_text():
                        txts['legend'].append(t)

        for t in itertools.chain(txts['xticks'], txts['yticks'], txts['legend']):
            #  Texts in these regions must not be set manually by the code.
            if 'Text-' in t.get_text():
                return False

        #  Check for overlaps
        for txt_elems in [txts['xticks'], txts['yticks'], txts['legend']]:
            for t_i, t_j in itertools.combinations(txt_elems, 2):
                if t_i.get_window_extent().overlaps(t_j.get_window_extent()):
                    return False

    return True


def _fig_serializer(fig):
    png = serialize_fig(fig, format='png', tight=True)
    if not check_rules(fig):
        return None
    return png


def synthesize_worker(viz_functions_queue: multiprocessing.Queue,
                      results_queue: multiprocessing.Queue,
                      query: Query,
                      instantiator: BaseInstantiator,
                      df_var_names: List[str]):
    while not viz_functions_queue.empty():
        viz_function = viz_functions_queue.get(block=True)

        sys.stdout.flush()
        for result in instantiator.instantiate(query, [viz_function], serializer=_fig_serializer):
            png = result['serialized']  # In bytes
            df_args = result['df_args_mapping']
            col_args = result['col_args_mapping']
            code = viz_function['code']
            func_call_args = ", ".join([*(f"{k}={df_var_names[v]}" for k, v in df_args.items()),
                                        *(f"{k}={v!r}" for k, v in col_args.items())])
            code = code + f"\n\nvisualization({func_call_args})"
            code_html = get_html(code)

            encoded_png = base64.b64encode(png).decode('utf-8')
            results_queue.put({
                'png': encoded_png,
                'code': code,
                'code_html': code_html,
            })


@attr.s(cmp=False, repr=False)
class SynthesisTask:
    query: Query = attr.ib()
    viz_functions: List[Dict] = attr.ib()
    instantiator: BaseInstantiator = attr.ib()
    callback: Callable = attr.ib()
    df_var_names: List[str] = attr.ib()

    polling_time: int = attr.ib(default=2)  # in seconds

    _active: bool = attr.ib(init=False, default=True)

    #  Workers
    _viz_functions_queue = attr.ib(init=False)
    _results_queue = attr.ib(init=False)
    _synthesis_worker = attr.ib(init=False)
    _polling_worker = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.setup()

    def setup(self):
        self._viz_functions_queue = multiprocessing.Queue()
        self._results_queue = multiprocessing.Queue()
        for viz_function in self.viz_functions:
            self._viz_functions_queue.put(obj=viz_function, block=True)

        self._synthesis_worker = multiprocessing.Process(target=synthesize_worker,
                                                         args=(self._viz_functions_queue,
                                                               self._results_queue,
                                                               self.query,
                                                               self.instantiator,
                                                               self.df_var_names))
        self._polling_worker = threading.Thread(target=self.polling_func)

    def start(self):
        self._synthesis_worker.start()
        self._polling_worker.start()

    def is_active(self):
        return self._active

    def terminate(self):
        self._active = False
        self._synthesis_worker.kill()

    def polling_func(self):
        while self._active:
            items = []
            while not self._results_queue.empty():
                items.append(self._results_queue.get())

            self.callback(items)
            if not self._synthesis_worker.is_alive():
                self._active = False
                break

            time.sleep(self.polling_time)

        self.callback([])


@attr.s(cmp=False, repr=False)
class App:
    searcher: BaseSearcher = attr.ib()
    instantiator: BaseInstantiator = attr.ib()
    dataframes: List[pd.DataFrame] = attr.ib()
    columns: List[Union[int, str]] = attr.ib()
    df_var_names: List[str] = attr.ib()

    #  Search bar for searching visualizations by text
    _viz_search_bar = attr.ib(init=False)

    #  Page Navigation
    _prev_page_button = attr.ib(init=False)
    _next_page_button = attr.ib(init=False)
    _page_navigation = attr.ib(init=False)
    _page_no: int = attr.ib(init=False, default=1)
    _page_size: int = attr.ib(init=False, default=10)

    #  Zoom bar for tiled visualization display
    _per_page_zoom_slider = attr.ib(init=False)
    _zoom_level: int = attr.ib(init=False, default=3)

    #  Main display widget for visualizations
    _status = attr.ib(init=False)
    _viz_display = attr.ib(init=False)

    #  Current Synthesis Task
    _current_task: SynthesisTask = attr.ib(init=False, default=None)
    _results: List[Dict] = attr.ib(init=False, factory=list)
    _seen_figures: Dict[str, int] = attr.ib(init=False, factory=dict)
    _idx_ctr: int = attr.ib(init=False, default=0)

    _cancel_button = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.build()

    def build(self):
        self._viz_search_bar = widgets.Text(
            placeholder='Type keywords to synthesize visualizations',
            description='Search',
            layout=Layout(width='auto'),
            disabled=False
        )

        self._prev_page_button = create_expanded_button('', '', icon='arrow-left')
        self._next_page_button = create_expanded_button('', '', icon='arrow-right')
        self._per_page_zoom_slider = widgets.IntSlider(
            value=self._zoom_level,
            min=1,
            max=8,
            step=1,
            description='Zoom:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self._cancel_button = create_expanded_button('Stop', 'danger')
        self._page_navigation = widgets.HBox([self._prev_page_button, self._next_page_button,
                                              self._per_page_zoom_slider, self._cancel_button])

        self._status = widgets.Output()
        self._viz_display = VizSynthesisWidget()
        self._viz_display.set_selection_callback(self.selection_callback)

        self._viz_search_bar.on_submit(self.onsubmit_search)
        self._prev_page_button.on_click(self.onclick_prev_page)
        self._next_page_button.on_click(self.onclick_next_page)
        self._per_page_zoom_slider.observe(self.on_zoom_change, 'value')
        self._cancel_button.on_click(self.onclick_cancel)

    def onsubmit_search(self, *args, **kwargs):
        text = self._viz_search_bar.value
        query = Query(
            query_str=text,
            provided_dfs=list(self.dataframes),
            requested_cols=list(self.columns),
        )

        viz_functions = self.searcher.search(query)
        viz_functions = [t for t in viz_functions if t['reusable']]

        self._results.clear()
        self._seen_figures.clear()
        if self._current_task is not None:
            self._current_task.terminate()

        self._current_task = SynthesisTask(query=query,
                                           viz_functions=viz_functions,
                                           instantiator=self.instantiator,
                                           callback=self.on_new_results,
                                           df_var_names=self.df_var_names)
        self.reset_display()
        self.update_display()
        self._current_task.start()

    def onclick_cancel(self, *args, **kwargs):
        if self._current_task is not None:
            self._current_task.terminate()

    def on_new_results(self, new_results: List[Dict]):
        for i in new_results:
            if i['png'] in self._seen_figures:
                item = self._results[self._seen_figures[i['png']]]
                if i['code'] not in item['code']:
                    item['code'].append(i['code'])
                    item['code_html'].append(i['code_html'])

            else:
                idx = self._idx_ctr
                self._idx_ctr += 1
                self._seen_figures[i['png']] = idx
                self._results.append({
                    **i,
                    'idx': idx,
                    'code': [i['code']],
                    'code_html': [i['code_html']],
                })

        self.update_display()
        if len(self._results) >= 100:
            self._current_task.terminate()

    def selection_callback(self, idx: int):
        item = self._results[idx]
        create_code_cell(item['code'][0])

    def reset_display(self):
        self._viz_display.clear()
        self._page_no = 1

    def update_display(self):
        p_no = self._page_no
        p_size = self._page_size
        num_results = len(self._results)
        total_pages = (num_results + (p_size - 1)) // p_size
        to_display = self._results[(p_no - 1) * p_size: p_no * p_size]

        if {i['idx'] for i in to_display} != {i['idx'] for i in self._viz_display.data}:
            self._viz_display.data = to_display

        self._page_navigation.layout.display = ''

        if p_no == 1:
            self._prev_page_button.disabled = True
        else:
            self._prev_page_button.disabled = False

        if p_no == total_pages:
            self._next_page_button.disabled = True
        else:
            self._next_page_button.disabled = False

        self.update_status()

    def update_status(self):
        self._status.clear_output()
        if self._current_task.is_active():
            if len(self._results) == 0:
                with self._status:
                    print("Searching...")
            else:
                with self._status:
                    print(f"Searching... Found {len(self._results)} visualizations so far.")

        else:
            if len(self._results) == 0:
                with self._status:
                    print("No visualizations found.")

            else:
                with self._status:
                    print(f"Found {len(self._results)} visualizations.")

    def onclick_prev_page(self, *args, **kwargs):
        self._page_no = max(self._page_no - 1, 0)
        self.update_display()

    def onclick_next_page(self, *args, **kwargs):
        p_size = self._page_size
        num_results = len(self._results)
        total_pages = (num_results + (p_size - 1)) // p_size
        self._page_no = min(self._page_no + 1, total_pages)
        self.update_display()

    def on_zoom_change(self, *args, **kwargs):
        self._zoom_level = self._per_page_zoom_slider.value
        self._viz_display.num_cols = self._zoom_level

    def display(self):
        self._page_navigation.layout.display = 'none'
        display(self._viz_search_bar)
        display(self._page_navigation)
        display(self._status)
        display(self._viz_display)

    def __del__(self):
        if self._current_task is not None:
            self._current_task.terminate()


def get_searcher(searcher_type: str):
    if searcher_type in _searcher_cache:
        return _searcher_cache[searcher_type]

    path_viz_functions = f"{common.PROJECT_DIR}/visualization_functions.pkl"
    if not os.path.exists(path_viz_functions):
        raise FileNotFoundError(f"File {path_viz_functions} not found.")

    with open(path_viz_functions, 'rb') as f:
        viz_functions = pickle.load(f)

    if searcher_type == 'simple-code':
        searcher = SimpleCodeSearcher(viz_functions)
    elif searcher_type == 'nl':
        searcher = NaturalLanguageSearcher(viz_functions)
    elif searcher_type == 'nl+code':
        searcher = WhooshNLPlusCodeSearcher(viz_functions)
    else:
        raise ValueError("Arg `searcher_type` must be one of ('simple-code', 'nl', 'nl+code')")

    _searcher_cache[searcher_type] = searcher
    return searcher


def get_instantiator(instantiator_type: str):
    if instantiator_type == 'simple-instantiator':
        return SimpleInstantiator()
    else:
        raise ValueError("Arg `instantiator_type` must be one of ('simple-instantiator', 'generality-instantiator').")


def synthesize(dfs: List[pd.DataFrame],
               columns: List[str],
               searcher_type: str = 'nl+code',
               instantiator_type: str = 'simple-instantiator'):
    searcher = get_searcher(searcher_type)
    instantiator = get_instantiator(instantiator_type)
    main_module = sys.modules["__main__"]
    var_names = []
    for df in dfs:
        cand_vars = []
        for k, v in main_module.__dict__.items():
            if v is df:
                cand_vars.append(k)

        if all(i.startswith("_") for i in cand_vars):
            var_names.append(cand_vars[0])
        else:
            var_names.append(next(i for i in reversed(cand_vars) if not i.startswith("_")))

    app = App(searcher=searcher,
              instantiator=instantiator,
              dataframes=dfs,
              columns=columns,
              df_var_names=var_names)
    app.build()
    app.display()
