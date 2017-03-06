
from __future__ import print_function, absolute_import


def assert_fails(fun, err, *args, **kwargs):
    """
    Assert that a function fails with a defined error.

    :param fun: the callable
    :param err: the error type
    :param args: the args for the function
    :param kwargs: keyword args for the function
    :return: True if failed, else raises Exception
    """
    try:
        fun(*args, **kwargs)
    except err:
        return True
    except Exception:
        raise
