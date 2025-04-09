from .base_reader import DesignReader
from .lefdef_reader import LefDefReader
from .hx_tmp_reader import HxTmpReader

def get_design_reader(design_name) -> type[DesignReader]:
    if design_name == 'blackparrot':
        return HxTmpReader
    else:
        return LefDefReader
