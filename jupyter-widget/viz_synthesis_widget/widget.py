import ipywidgets as widgets
from traitlets import Unicode, List, Int


@widgets.register
class VizSynthesisWidget(widgets.DOMWidget):
    """
    Python-side Code for the Synthesis Widget.

    The `data` attribute should be a list of records (dictionaries).
    Every record should have at least the following entries:
    {
    'png': The png corresponding to the visualization in base64 format,
    'code_html': HTML representation of the code. You can use the pygments library on the Python side to format code,
    'idx': An integer uniquely identifying the element. This must be unique across all elements across all pages.
    }

    The `num_cols` attribute controls the number of columns of the grid to display the elements in.
    Modifying this attribute directly will simultaneously update the grid.
    """
    _view_name = Unicode('VizSynthesisWidgetView').tag(sync=True)
    _model_name = Unicode('VizSynthesisWidgetModel').tag(sync=True)
    _view_module = Unicode('viz_synthesis_widget').tag(sync=True)
    _model_module = Unicode('viz_synthesis_widget').tag(sync=True)
    _view_module_version = Unicode('^0.1.0').tag(sync=True)
    _model_module_version = Unicode('^0.1.0').tag(sync=True)
    data = List(default_value=[]).tag(sync=True)
    num_cols = Int(3).tag(sync=True)
    selection = Int(-1).tag(sync=True)

    def clear(self):
        self.data = []
        self.num_cols = 3
        self.selection = -1

    def set_selection_callback(self, callback):
        """
        When the Select button is clicked on the javascript side, the passed callback will be triggered.
        The argument passed will be the 'idx' entry corresponding to the data item selected.
        """
        def wrapped_callback(*args, **kwars):
            if self.selection != -1:
                callback(self.selection)

        self.observe(wrapped_callback, 'selection')
