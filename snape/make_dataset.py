
########################################################################
# Creates a Machine Learning Problem Dataset
# Suitable for assigning to a student for a homework assignment
# Mostly convenience code for sklearn's make_classification routines
#
#
########################################################################

from __future__ import print_function, absolute_import, division
from sklearn.datasets import make_classification, make_regression
from sklearn.externals import six
from snape.utils import assert_is_type, get_random_state, assert_valid_percent
import pandas as pd
import numpy as np
import argparse
import json
import re
import os
import sys

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

try:
    long
except NameError:  # python 3
    long = int


def parse_args(args):
    """
    Returns arguments passed at the command line as a dict
    """
    parser = argparse.ArgumentParser(description='Generates a machine Learning Dataset.')
    parser.add_argument('-c', help="Config File Location", required=True,
                        dest='config')
    return vars(parser.parse_args(args))


def load_config(config_name):
    """
    Loads a json config file and returns a config dictionary.
    :param config_name: the path to the config json
    """
    with open(config_name) as config_file:
        config = json.load(config_file)
        return config


def rename_columns(df, prefix='x'):
    """
    Rename the columns of a dataframe to have X in front of them

    :param df: data frame we're operating on
    :param prefix: the prefix string
    """
    # the prefix needs to be a string
    assert_is_type(prefix, six.string_types)  # 2 & 3 compatible

    df = df.copy()
    df.columns = [prefix + str(i) for i in df.columns]
    return df


def insert_missing_values(df, percent_rows, random_state=None):
    """
    Inserts missing values into a data frame.

    :param df: data frame we're operating on
    :param percent_rows: the percentage of rows that should have a missing value.
    :param random_state: the numpy RandomState
    :return: a df with missing values
    """
    # get the initialized random_state (if not already initialized)
    random_state = get_random_state(random_state)
    df = df.copy()

    def _insert_random_null(x):
        """
        Chose a random column in a df row to null. This
        operates in-place. But it's on the copy, so it should be OK.

        :param x: the data frame
        """
        # -1 because last col will always be y
        x[random_state.randint(0, len(x) - 1)] = np.nan
        return x

    # this is a "truthy" check. If it's zero or False, this will work.
    if not percent_rows:
        return df
    else:
        # otherwise validate that it's a float
        percent_rows = assert_valid_percent(percent_rows, eq_upper=True)  # eq_lower not necessary because != 0.
        sample_index = df.sample(frac=percent_rows, random_state=random_state).index  # random sample of rows to null
        df.loc[sample_index] = df.loc[sample_index].apply(_insert_random_null, axis=1)
        return df


def insert_special_char(character, df, random_state=None):
    """
    Chooses a column to reformat as currency or percentage, including a $ or % string, to make cleaning harder

    :param character: either $ or %
    :param df: the dataframe we're operating on
    :param random_state: the numpy RandomState
    :return: A dataframe with a single column chosen at random converted to a % or $ format
    """
    # get the initialized random_state (if not already initialized)
    random_state = get_random_state(random_state)
    df = df.copy()

    # choose a column at random, that isn't Y.  Only choose from numeric columns (no other eviled up columns)
    chosen_col = random_state.choice([col for col in df.select_dtypes(include=['number']).columns if col != 'y'])

    # assert that character is a string and that it's in ('$', '%')
    assert_is_type(character, six.string_types)
    if character not in ('$', '%'):
        raise ValueError('expected `character` to be in ("$", "%"), but got {0}'.format(character))

    # do scaling first:
    df[chosen_col] = (df[chosen_col] - df[chosen_col].mean()) / df[chosen_col].std()

    # do the specific div/mul operations
    if character is "$":
        # multiply by 1000, finally add a $
        df[chosen_col] = (df[chosen_col] * 1000).round(decimals=2).map(lambda x: "$" + str(x))
    else:  # elif character is "%":
        # divide by 100, finally add a $
        df[chosen_col] = (df[chosen_col] / 100).round(decimals=2).map(lambda x: str(x) + "%")

    return df


def create_categorical_features(df, label_list, random_state=None):
    """
    Creates random categorical variables

    :param df: data frame we're operation on
    :param label_list: A list of lists, each list is the labels for one categorical variable
    :param random_state: the numpy RandomState
    :return: A modified dataframe

    Example:

    create_categorical_features(df, [['a','b'], ['red','blue']])

    """
    random_state = get_random_state(random_state)

    df = df.copy()
    n_categorical = len(label_list)

    # get numeric columns ONCE so we don't have to do it every time we loop:
    numer_cols = [col for col in df.select_dtypes(include=['number']).columns if col != 'y']

    for i in range(0, n_categorical):
        # we might be out of numerical columns!
        if not numer_cols:
            break

        # chose a random numeric column that isn't y
        chosen_col = random_state.choice(numer_cols)
        # pop the chosen_col out of the numer_cols
        numer_cols.pop(numer_cols.index(chosen_col))

        # use cut to convert that column to categorical
        df[chosen_col] = pd.cut(df[chosen_col], bins=len(label_list[i]), labels=label_list[i])

    return df


def create_classification_dataset(n_samples, n_features, n_informative, n_redundant, n_repeated,
                                  n_clusters_per_class, weights, n_classes, random_state=None):
    """
    Creates a binary classifier dataset

    :param n_samples: number of observations
    :param n_features: number of  features
    :param n_informative: number of informative features
    :param n_redundant: number of multicolinear
    :param n_repeated:  number of perfect collinear features
    :param n_clusters_per_class:  gaussian clusters per class
    :param weights: list of class balances, e.g. [.5, .5]
    :param n_classes: the number of class levels
    :param random_state: the numpy RandomState
    :return: the requested dataframe
    """
    random_state = get_random_state(random_state)
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated,
                               n_clusters_per_class=n_clusters_per_class, weights=weights,
                               scale=(np.random.rand(n_features) * 10), n_classes=n_classes,
                               random_state=random_state)
    # cast to a data frame
    df = pd.DataFrame(X)
    # rename X columns
    df = rename_columns(df)
    # and add the Y
    df['y'] = y
    return df


def create_regression_dataset(n_samples, n_features, n_informative, effective_rank, tail_strength,
                              noise, random_state=None):
    """
    Creates a regression dataset

    :param n_samples: number of observations
    :param n_features: number of features
    :param n_informative: number of informative features
    :param n_targets: The number of regression targets, i.e., the dimension of the y output vector associated with a sample. By default, the output is a scalar.
    :param effective_rank: approximate number of singular vectors required to explain data
    :param tail_strength: relative importance of the fat noisy tail of the singular values profile
    :param noise: standard deviation of the gaussian noise applied to the output
    :param random_state: the numpy RandomState
    :return: the requested dataframe
    """
    random_state = get_random_state(random_state)
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           n_targets=1, effective_rank=effective_rank, tail_strength=tail_strength,
                           noise=noise, random_state=random_state)

    # cast to a data frame
    df = pd.DataFrame(X)
    # rename X columns
    df = rename_columns(df)
    # and add the Y
    df['y'] = y
    return df


def make_star_schema(df, out_path="." + os.path.sep):
    """
    Converts dataset to star-schema fact and dimension tables. Dimension tables are written out to CSV files,
    and the dataframe passed to the function is converted into a 'fact' table and returned as a dataframe (this
    file is NOT written out at this point because the fact table would be subject to test/train split functions,
    and dimension tables would not be). 

    :param df: Source dataframe
    :param out_path: path to write the dimension files to
    :return: dataframe with dimension table
    """
    def _get_categorical_columns(x):  # don't shadow df from outer scope
        return x.select_dtypes(include=['category', 'object']).columns

    def _find_dollars(text):
        return 1 if re.match(r'^\$-?\d+\.?\d+', str(text)) else 0

    def _find_percentages(text):
        return 1 if re.search(r'^-?\d+\.?\d+[%]$', str(text)) else 0

    def _is_special_char(list_object):
        if list_object.dtype != 'O':
            return False
        else:
            percent_sum = sum(list_object.apply(_find_percentages))
            dollars_sum = sum(list_object.apply(_find_dollars))

            return (percent_sum / list_object.count() == 1) or (dollars_sum / list_object.count() == 1)

    # Get the categorical columns
    cols = _get_categorical_columns(df)
    assert len(cols) > 0, "No categorical variables exist in this dataset; star schema cannot be developed."
    
    # Iterate through the categorical columns
    for cat_column in cols:
        
        # Determine if the list includes requested entropy or not (NOTE: Decided not to make dimension 
        # tables before this command so dimension keys CAN'T be selected for entropy)
        if not _is_special_char(df[cat_column]):  # previously was "is not True" but not very pythonic
            
            # Turn the value counts into a dataframe
            vals = pd.DataFrame(df[cat_column].value_counts())

            # todo: Sara, the following seems hacky... is there a better way to do this?
            # Reset the index to add index as the key
            vals.reset_index(inplace=True)  # Puts the field names into the dataframe
            vals.reset_index(inplace=True)  # Puts the index numbers in as integers
            
            # Name the column with the same name as the column 'value_count'
            vals.rename(index=str,
                        columns={'level_0': 'primary_key',
                                 'index': 'item',
                                 cat_column: 'value_count'
                                 },
                        inplace=True)
            
            # Make a df out of just the value and the mapping
            val_df = vals[['primary_key', 'item']]

            # todo: Sara, this is hacky (but really cool!) Could you please write a comment block
            # todo: ... explaining exactly what you're achieving here?
            # Make a dimension df by appending a NaN placeholder
            val_df.item.cat.add_categories('Not specified', inplace=True)
            val_df = val_df.append({'primary_key': -1, 'item': 'Not specified'}, ignore_index=True)

            # todo: Sara, should we take another param in this function that can either
            # todo: ... permit or prevent accidentally overwriting an existing file?
            # Write the new dimension table out to CSV
            dim_file_name = cat_column + '_dim.csv'
            val_df.to_csv(out_path + dim_file_name, index=False)
            
            # Set the index up for mapping
            val_df.set_index('item', inplace=True)
            
            # Convert to dict for mapping
            mapper = val_df.to_dict().get('primary_key')
            
            # Fill the NaNs in the dataframe's categorical column to 'Not Specified'
            df[cat_column].cat.add_categories('Not specified', inplace=True)
            df[cat_column].fillna('Not specified', inplace=True)
            
            # Insert new column into the dataframe
            df.insert(df.shape[1], cat_column + '_key', df[cat_column].map(mapper))

            # Drop cat column from the dataframe
            df.drop(cat_column, axis=1, inplace=True)
            
    # Now, reset the dataframe's index and rename the index column as 'primary_key'
    df.reset_index(inplace=True)
    df_cols = df.columns
    df_cols = df_cols.delete(0)
    df_cols = df_cols.insert(0, 'primary_key')
    df.columns = df_cols

    # Return the main dataframe as a 'fact' table, which will then be split into test/train splits
    # dimension tables are immune to this
    return df.copy()


def write_dataset(df, file_name, out_path="." + os.path.sep):
    """
    Writes generated dataset to file

    :param df: dataframe to write
    :param file_name: beginning of filename
    :param out_path: the path to write the dataset
    :return: None
    """
    # todo: Mike, do we want to take a param for overwriting existing files?
    df_train, df_testkey = train_test_split(df, test_size=.2)

    df_train.to_csv(out_path + file_name + "_train.csv", index=False)
    df_test = df_testkey.drop(['y'], axis=1)
    df_test.to_csv(out_path + file_name + "_test.csv", index=False)
    df_testkey.to_csv(out_path + file_name + "_testkey.csv", index=False)


def make_dataset(config=None):
    """
    Creates a machine learning dataset based on command line arguments passed

    :param config: a configuration dictionary, or None if called from the command line
    :return: None
    """

    if config is None:
        # called from the command line so parse configuration
        args = parse_args(sys.argv[1:])
        config = load_config(args['config'])

    print('-' * 80)
    c_type = config['type']  # avoid multiple lookups - fails with key error if not present
    if c_type not in ('regression', 'classification'):
        raise ValueError('type must be in ("regression", "classification"), but got %s' % c_type)
    reg = c_type == 'regression'

    # get defaults - these are the defaults from sklearn.
    def _safe_get_with_default(cfg, key, default):
        if key not in cfg:
            print("Warning: %s not in configuration, defaulting to %r" % (key, default))
            return default
        return cfg[key]

    n_samples = _safe_get_with_default(config, 'n_samples', 100)
    n_features = _safe_get_with_default(config, 'n_features', 20 if not reg else 100)  # diff defaults in sklearn
    n_informative = _safe_get_with_default(config, 'n_informative', 2 if not reg else 10)  # diff defaults in sklearn
    n_redundant = _safe_get_with_default(config, 'n_redundant', 2)
    n_repeated = _safe_get_with_default(config, 'n_repeated', 0)
    n_clusters_per_class = _safe_get_with_default(config, 'n_clusters_per_class', 2)
    weights = _safe_get_with_default(config, 'weights', None)
    n_classes = _safe_get_with_default(config, 'n_classes', 2)
    effective_rank = _safe_get_with_default(config, 'effective_rank', None)
    tail_strength = _safe_get_with_default(config, 'tail_strength', 0.5)
    noise = _safe_get_with_default(config, 'noise', 0.)
    seed = _safe_get_with_default(config, 'random_seed', 42)

    # get the random state
    random_state = get_random_state(seed)

    # create the base dataset
    if not reg:
        print('Creating Classification Dataset...')
        df = create_classification_dataset(n_samples=n_samples, n_features=n_features,
                                           n_informative=n_informative, n_redundant=n_redundant,
                                           n_repeated=n_repeated, n_clusters_per_class=n_clusters_per_class,
                                           weights=weights, n_classes=n_classes, random_state=random_state)

    else:  # elif c_type == 'regression':
        print('Creating Regression Dataset...')
        df = create_regression_dataset(n_samples=n_samples, n_features=n_features,
                                       n_informative=n_informative, effective_rank=effective_rank,
                                       tail_strength=tail_strength, noise=noise, random_state=random_state)

    # make sure to use safe lookups to avoid KeyErrors!!!
    label_list = _safe_get_with_default(config, 'label_list', None)
    do_categorical = label_list is not None and len(label_list) > 0

    if do_categorical:
        print("Creating Categorical Features...")

        df = create_categorical_features(df, label_list, random_state=random_state)

    # insert entropy
    insert_dollar = _safe_get_with_default(config, 'insert_dollar', "No")
    insert_percent = _safe_get_with_default(config, 'insert_percent', "No")

    if any(entropy == "Yes" for entropy in (insert_dollar, insert_percent)):
        print("Inserting Requested Entropy...")

        # add $ or % column if requested
        if insert_dollar == "Yes":
            df = insert_special_char('$', df, random_state=random_state)
        if insert_percent == "Yes":
            df = insert_special_char('%', df, random_state=random_state)
    
    # insert missing values
    pct_missing = _safe_get_with_default(config, 'pct_missing', None)
    df = insert_missing_values(df, pct_missing, random_state=random_state)
    
    # Convert dataset to star schema if requested
    star_schema = _safe_get_with_default(config, 'star_schema', "No")
    outpath = _safe_get_with_default(config, 'out_path', "." + os.path.sep)
    if star_schema == "Yes":
        # Check the number of categorical variables
        if do_categorical:
            df = make_star_schema(df, outpath)
        else:
            print("No categorical variables added. Dataset cannot be transformed into a star schema. "
                  "Dataset will be generated as a single-table dataset...")

    print("Writing Train/Test Datasets")
    write_dataset(df, _safe_get_with_default(config, 'output', 'my_dataset'), outpath)


if __name__ == "__main__":
    make_dataset()
