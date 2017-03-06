
from __future__ import print_function, absolute_import, division
import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from snape.score_dataset import *
from snape.utils import get_random_state, assert_valid_percent
import os
from nose.tools import with_setup

random_state = get_random_state(42)
y_rand = (random_state.rand(200))

r = {'y': y_rand * 10,
     'y_hat': y_rand * 10 - y_rand
     }

regression_df = pd.DataFrame(r)
c = {'y': [1, 1, 1, 1, 0, 0, 0, 0],
     'y_hat': [1, 0.9, 0.4, 0.95, 0, 0.1, 0.6, 0.15]
     }

classification_df = pd.DataFrame(c)
m = {'y': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
     'y_hat': [0, 1, 0, 1, 1, 3, 1, 2, 2, 3, 2, 3]
     }

multiclass_df = pd.DataFrame(m)


def test_guess_problem_type():
    assert guess_problem_type(regression_df['y']), 'regression'
    assert guess_problem_type(classification_df['y']), 'binary'
    assert guess_problem_type(multiclass_df['y']), 'multiclass'


def test_score_binary_classification():
    y = classification_df['y']
    y_hat = classification_df['y_hat']
    assert score_binary_classification(y, y_hat, report=False)[0] == 0.9375
    assert "---Binary Classification Score---" in score_binary_classification(y, y_hat, report=False)[1]


def test_score_multiclass_classification():
    y = multiclass_df['y']
    y_hat = multiclass_df['y_hat']
    assert round(score_multiclass_classification(y, y_hat, report=False)[0], 2) == 0.67
    assert "---Multiclass Classification Score---" in score_multiclass_classification(y, y_hat, report=False)[1]


def test_score_regression():
    y = regression_df['y']
    y_hat = regression_df['y_hat']
    assert_almost_equal(round(score_regression(y, y_hat, report=False)[0], 2), 0.48)
    assert "---Regression Score---" in score_regression(y, y_hat, report=False)[1]


def test_arg_parser():
    args = parse_args(["-ktest_key.csv","-ptest_pred.csv"])
    assert args['pred'] == 'test_pred.csv', "parse_args failed to parse it's argument"
    assert args['key'] == 'test_key.csv', "parse_args failed to parse it's argument"


def setup_file_func():
    y = classification_df['y']
    y_hat = classification_df['y_hat']
    y.to_csv("y_test.csv", header=True, index=False)
    y_hat.to_csv("y_hat_test.csv", header=False, index=False)


def teardown_file_func():
    os.remove("y_test.csv")
    os.remove("y_hat_test.csv")


@with_setup(setup=setup_file_func, teardown=teardown_file_func)
def test_read_files():
    y, y_hat = read_files("y_test.csv", "y_hat_test.csv")
    assert y.shape[0] > 1, "y should have more than one row"
    assert y_hat.shape[0] > 1, "y hat should have more than one row"


@with_setup(setup=setup_file_func, teardown=teardown_file_func)
def test_score_dataset():
    results = score_dataset(y_file="y_test.csv", y_hat_file="y_hat_test.csv")
    assert_valid_percent(results[0], "Not a valid percent for AUC")




