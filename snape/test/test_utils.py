
from snape.utils import assert_valid_percent, get_random_state
import pytest


@pytest.mark.parametrize(
    'pct,kwargs', [

        pytest.param(
            0.5,
            {},
        ),

        pytest.param(
            0.9,
            {},
        ),

        pytest.param(
            0.1,
            {},
        ),

        # these only pass with certain kwargs
        pytest.param(
            0.0,
            {"eq_lower": True},
        ),

        pytest.param(
            1.0,
            {"eq_upper": True},
        ),

    ]
)
def test_valid_percentages(pct, kwargs):
    assert_valid_percent(pct, **kwargs)


@pytest.mark.parametrize('pct', [0.0, 1.0])
def test_invalid_percetages(pct):
    with pytest.raises(ValueError):
        assert_valid_percent(x=pct)


@pytest.mark.parametrize(
    'x', [
        'some random string',
        {'an': 'iterable'}
    ]
)
def test_random_state_fails(x):
    with pytest.raises(TypeError):
        get_random_state(x)
