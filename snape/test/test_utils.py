
from __future__ import print_function, absolute_import, division
from snape.utils import assert_valid_percent, get_random_state
from snape.testing import assert_fails


def test_valid_percentages():
    # these are valid:
    assert_valid_percent(0.5)
    assert_valid_percent(0.9)
    assert_valid_percent(0.1)

    # these will fail:
    assert_fails(assert_valid_percent, ValueError, x=0.0)
    assert_fails(assert_valid_percent, ValueError, x=1.0)

    # these will pass:
    assert_valid_percent(x=1.0, eq_upper=True)
    assert_valid_percent(x=0.0, eq_lower=True)


def test_random_state():
    assert_fails(get_random_state, TypeError, 'some random string')


def tests_assert_fails():
    # meta tests...
    def _fails_with_assertion_error():
        def _non_failing_func():
            return None
        return assert_fails(_non_failing_func, ValueError)

    # show that it will raise AssertionError if it does not fail with expected Exception
    assert_fails(_fails_with_assertion_error, AssertionError,)
