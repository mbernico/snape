
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
    except err:  # except the EXPECTED error
        return True
    except Exception:  # except a DIFFERENT error
        pass

    # if it gets here, it either raise another exception or didn't fail:
    raise AssertionError('Expected function to fail with %s' % type(err))
