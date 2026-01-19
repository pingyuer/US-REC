import os
import importlib

__all__ = []


_pkg_dir = os.path.dirname(__file__)

for fname in os.listdir(_pkg_dir):
    if fname.endswith(".py") and not fname.startswith("_") and fname != "__init__.py":
        module_name = f"{__name__}.{fname[:-3]}"
        module = importlib.import_module(module_name)

        if hasattr(module, "__all__"):
            globals().update({k: getattr(module, k) for k in module.__all__})
            __all__.extend(module.__all__)