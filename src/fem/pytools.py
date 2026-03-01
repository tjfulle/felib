from functools import wraps


def _require_unfrozen(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_frozen", False):
            class_name = type(self).__name__
            method_name = method.__name__
            raise RuntimeError(
                f"'{class_name}.{method_name}' cannot be called after {class_name} is frozen"
            )
        return method(self, *args, **kwargs)

    return wrapper


def frozen_property(method):
    @property
    @wraps(method)
    def wrapper(self):
        if not getattr(self, "_frozen", False):
            class_name = type(self).__name__
            method_name = method.__name__
            raise RuntimeError(
                f"'{class_name}.{method_name}' cannot be called until after {class_name} is frozen"
            )
        return method(self)

    return wrapper
