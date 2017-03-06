
from __future__ import print_function, absolute_import, division
import pandas as pd
from snape.make_dataset import *
from snape.utils import get_random_state
from nose.tools import assert_raises, with_setup
import glob
import os
import sys


random_state = get_random_state(42)


# move to nosetests instead of unit tests
def test_create_classification_dataset():
    df = create_classification_dataset(n_samples=100, n_features=10, n_informative=3, n_redundant=0,
                                       n_repeated=0, n_clusters_per_class=2, weights=[0.5, 0.5], n_classes=2,
                                       random_state=random_state)

    assert df.shape[0] == 100, "Sample Size Doesn't Match"
    assert df.shape[1] == 11, "Feature Count"
    assert df['y'].value_counts().shape[0] == 2, "Expected Shape of Classes Do Not Match"


def test_create_regression_dataset():
    df = create_regression_dataset(n_samples=100, n_features=10, n_informative=3, effective_rank=1,
                                   tail_strength=0.5, noise=0.0, random_state=random_state)

    assert df.shape[0] == 100, "Sample Size Doesn't Match"
    assert df.shape[1] == 11, "Feature Count"


def test_create_categorical_features():
    df = pd.DataFrame(random_state.randn(100, 4), columns=list('ABCD'))
    cat_df = create_categorical_features(df, [['a', 'b'], ['red', 'blue']], random_state=random_state)
    assert cat_df.dtypes.value_counts()['category'] == 2, 'Category levels'  # there should be 2 category variables


def test_insert_special_char():
    df = pd.DataFrame(random_state.randn(100, 1), columns=list('A'))
    df_spec = insert_special_char("$", df, random_state=random_state)
    assert df_spec['A'].str.contains('$').all()

    df_spec = insert_special_char("%", df, random_state=random_state)
    assert df_spec['A'].str.contains('$').all()

    # using a non $ or % should raise a value error
    assert_raises(ValueError, insert_special_char, "!", df, random_state=random_state)


def test_insert_missing_values():
    df = pd.DataFrame(random_state.randn(100, 4), columns=list('ABCD'))
    df_result = insert_missing_values(df, 1, random_state=random_state)
    assert df_result.isnull().any().any()

    df_result = insert_missing_values(df, 0, random_state=random_state)
    assert not df_result.isnull().any().any()


def test_star_schema():
    df = create_classification_dataset(n_samples=100, n_features=10, n_informative=3, n_redundant=0,
                                       n_repeated=0, n_clusters_per_class=2, weights=[0.5, 0.5], n_classes=2,
                                       random_state=random_state)

    df = create_categorical_features(df, [['a', 'b'], ['red', 'blue']], random_state=random_state)
    df = insert_special_char('$', df, random_state=random_state)
    df = insert_special_char('%', df, random_state=random_state)
    df = insert_missing_values(df, .8, random_state=random_state)
    fact_df = make_star_schema(df)

    # Assert file generation
    file_list = glob.glob('./*_dim.csv')
    diff_list = list(filter(lambda x: x.endswith('_dim.csv'), file_list))
    assert len(diff_list) == 2

    # Delete the tester files
    for file_path in file_list:
        os.remove(file_path)

    # Assert key column creation
    columns = fact_df.columns
    key_cols = list(filter(lambda x: x.endswith('_key'), columns))
    assert len(key_cols) == 3

    # Assert key columns don't contain any nulls
    key_df = fact_df[key_cols]
    na_df = key_df.dropna()
    assert len(na_df) == len(key_df), "Nulls exist in the dimension key columns in the star schema."

    # Assert that an index named 'primary_key' was added.
    assert 'primary_key' in fact_df.columns, "Index named pk was not added to the fact table"
    assert len(fact_df.primary_key.value_counts()) == len(fact_df), "Primary key isn't unique."


def test_load_json():
    jf = os.path.join(os.path.dirname(__file__), '../../example/config_classification.json')
    c = load_config(jf)
    assert c['type'] == 'classification', "JSON load not sane"


def test_arg_parser():
    args = parse_args(["-ctest.json"])
    assert args['config'] == 'test.json', "parse_args failed to parse it's argument"

def write_dataset_teardown_func():
    os.remove("test_test.csv")
    os.remove("test_testkey.csv")
    os.remove("test_train.csv")


@with_setup(setup=None, teardown=write_dataset_teardown_func)
def test_write_dataset():
    df = pd.DataFrame(random_state.randn(100, 5), columns=list('ABCDy'))
    write_dataset(df, "test")





