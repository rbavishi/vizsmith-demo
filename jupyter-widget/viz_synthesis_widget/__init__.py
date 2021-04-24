from .__meta__ import __version__

from .widget import *

def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'viz_synthesis_widget',
        'require': 'viz_synthesis_widget/extension'
    }]
