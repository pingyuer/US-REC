# trainers/hooks/registry.py
_HOOKS = {}

def register_hook(name):
    def decorator(cls):
        _HOOKS[name] = cls
        return cls
    return decorator

def build_hook(name, **kwargs):
    if name not in _HOOKS:
        raise KeyError(f"Hook '{name}' not found. Available: {list(_HOOKS.keys())}")
    return _HOOKS[name](**kwargs)
