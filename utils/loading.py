import torch
import sys
import importlib
def torch_load_legacy(path):
    """Load network with legacy environment."""

    # Setup legacy env (for older networks)
    #_setup_legacy_env()

    # Load network
    checkpoint_dict = torch.load(path, map_location='cpu')

    # Cleanup legacy
    #_cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('ltr')
    sys.modules['dlframework'] = sys.modules['ltr']
    sys.modules['dlframework.common'] = sys.modules['ltr']
    importlib.import_module('ltr.admin')
    sys.modules['dlframework.common.utils'] = sys.modules['ltr.admin']
    for m in ('model_constructor', 'stats', 'settings', 'local'):
        importlib.import_module('ltr.admin.' + m)
        sys.modules['dlframework.common.utils.' + m] = sys.modules['ltr.admin.' + m]


def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]