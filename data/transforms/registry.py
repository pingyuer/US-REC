# transforms/registry.py

import importlib

_TRANSFORMS = {}

_BUILTINS_IMPORTED = False


def _ensure_builtin_imports() -> None:
    global _BUILTINS_IMPORTED
    if _BUILTINS_IMPORTED:
        return
    _BUILTINS_IMPORTED = True
    # Best-effort imports so built-in transforms register themselves.
    for mod in (
        "data.transforms.finalize_ops",
        "data.transforms.tui_ops",
        "data.transforms.custom_transforms",
        "data.transforms.custon_transforms",  # legacy shim
    ):
        try:
            importlib.import_module(mod)
        except Exception:
            continue


def register_transform(name):
    """Decorator: register a custom transform."""
    def decorator(cls):
        _TRANSFORMS[name] = cls
        return cls
    return decorator

def get_transform_cls(name):
    _ensure_builtin_imports()
    if name in _TRANSFORMS:
        return _TRANSFORMS[name]
    # fallback: try torchvision v2
    try:
        tv_transforms = importlib.import_module("torchvision.transforms.v2")
        if hasattr(tv_transforms, name):
            return getattr(tv_transforms, name)
    except ImportError:
        pass

    raise KeyError(
        f"Transform '{name}' not found. "
        f"Available custom: {list(_TRANSFORMS.keys())}, "
        f"torchvision.v2: see https://pytorch.org/vision/stable/transforms.html"
    )
