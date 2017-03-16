
from __future__ import print_function, absolute_import, division
from snape.utils import assert_valid_percent, get_random_state
from nose.tools import assert_raises


def test_valid_percentages():
    # these are valid:
    assert_valid_percent(0.5)
    assert_valid_percent(0.9)
    assert_valid_percent(0.1)

    # these will fail:
    assert_raises(ValueError,assert_valid_percent, x=0.0)
    assert_raises(ValueError, assert_valid_percent, x=1.0)

    # these will pass:
    assert_valid_percent(x=1.0, eq_upper=True)
    assert_valid_percent(x=0.0, eq_lower=True)


def test_random_state():
    assert_raises(TypeError, get_random_state, 'some random string')

