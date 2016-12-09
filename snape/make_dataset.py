#!/usr/bin/env python3

########################################################################
# Creates a Machine Learning Problem Dataset
# Suitable for assigning to a student for a homework assignment
# Mostly convenience code for sklearn's make_classification routines
#
#
########################################################################

from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
import argparse
import json
import random


def parse_args():
    """
    Returns arguments passed at the command line as a dict

    :return: configuration dictionary

    """
    parser = argparse.ArgumentParser(description='Generates a machine Learning Dataset.')
    # parser.add_argument('-f', '--foo', help='Description for foo argument', required=True)
    parser.add_argument('-c', help="Config File Location", required=True,
                        dest='config')
    args = vars(parser.parse_args())
    return args


def load_config(config_name):
    """
    loads a json config file and returns a dictionary args

    :param config_name: name of json configuration file
    :return: config dict

    """
    with open(config_name) as config_file:
        config = json.load(config_file)
        return config


def rename_columns(df):
    """
    Rename the columns of a dataframe to have X in front of them
    :param df: dataframe we want to modify
    :return: altered dataframe

    """
    df = df.copy()
    col_names = ["x" + str(i) for i in df.columns]
    df.columns = col_names
    return df


def insert_missing_values(df, percent_rows):
    """
    Inserts missing values into a dataframe.

    :param df: dataframe we're operating on
    :param percent_rows: the percentage of rows that should have a missing value.
    :return: a df with missing values

    """

    def insert_random_null(x):
        """
        Chose a random column in a df row to null
        :param x: dataframe row
        :return: row with null

        """

        col = random.randint(0, len(x) - 2)  # -2 because last col will always be y
        x[col] = np.nan
        return x

    df = df.copy()

    if (percent_rows == 0) or (percent_rows is None):
        return df
    else:
        sample_index = df.sample(frac=percent_rows).index  # random sample of rows to null
        df.loc[sample_index] = df.loc[sample_index].apply(insert_random_null, axis=1)
        return df


def insert_special_char(character, df):
    """
    Chooses a column to reformat as currency or percentage, including a $ or % string, to make cleaning harder

    :param character: either $ or %
    :param df: the dataframe we're operating on
    :return: A dataframe with a single column chosen at random converted to a % or $ format

    """

    df = df.copy()
    # choose a column at random, that isn't Y.  Only choose from numeric columns (no other eviled up columns)
    chosen_col = random.choice([col for col in df.select_dtypes(include=['number']).columns if col != 'y'])

    if character is "$":
        # rescale the column to 0 mean, 1 std dev, then multiply by 1000
        # lastly, stick a $ in front of it
        df[chosen_col] = ((df[chosen_col] - df[chosen_col].mean()) / df[chosen_col].std() * 1000).round(decimals=2)\
            .map(lambda x: "$" + str(x))
        return df

    if character is "%":
        # rescale the column to 0 mean, 1 std dev, then divide by 100
        # lastly, stick a % after it
        df[chosen_col] = (((df[chosen_col] - df[chosen_col].mean()) / df[chosen_col].std()) / 100).round(decimals=2)\
                               .map(lambda x: str(x) + "%")
        return df


def create_categorical_features(df, n_categorical, label_list):
    """
    Creates random categorical variables

    :param n_categorical: The number of categorical variables to create
    :param label_list: A list of lists, each list is the labels for one categorical variable
    :return: A modified dataframe

    Example:

    create_categorical_features(df, 2, [['a','b'], ['red','blue']])

    """
    df = df.copy()
    try:
        len(label_list) != n_categorical
    except:
        "You must specify a label for every categorical"

    for i in range(0, n_categorical):
        # chose a random numeric column that isn't y
        chosen_col = random.choice([col for col in df.select_dtypes(include=['number']).columns if col != 'y'])
        # use cut to convert that column to categorical
        df[chosen_col] = pd.cut(df[chosen_col], bins=len(label_list[i]), labels=label_list[i])

    return df


def create_classification_dataset(n_samples, n_features, n_informative, n_redundant, n_repeated, n_clusters_per_class, weights, n_classes):
    """

    Creates a binary classifier dataset

    :param n_samples: number of observations
    :param n_features: number of  features
    :param n_informative: number of informative features
    :param n_redundant: number of multicolinear
    :param n_repeated:  number of perfect collinear features
    :param n_clusters_per_class:  gaussian clusters per class
    :param weights: list of class balances, e.g. [.5, .5]
    :return: the requested dataframe

    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                               n_redundant=n_redundant, n_repeated=n_repeated,
                               n_clusters_per_class=n_clusters_per_class, weights=weights,
                               scale=(np.random.rand(n_features) * 10), n_classes=n_classes)
    # cast to a data frame
    df = pd.DataFrame(X)
    # rename X columns
    df = rename_columns(df)
    # and add the Y
    df['y'] = y
    return df


def create_regression_dataset(n_samples, n_features, n_informative, effective_rank, tail_strength, noise):
    """

    Creates a regression dataset

    :param n_samples: number of observations
    :param n_features: number of features
    :param n_informative: number of informative features
    :param n_targets: The number of regression targets, i.e., the dimension of the y output vector associated with a sample. By default, the output is a scalar.
    :param effective_rank: approximate number of singular vectors required to explain data
    :param tail_strength: relative importance of the fat noisy tail of the singular values profile
    :param noise: standard deviation of the gaussian noise applied to the output
    :return: the requested dataframe
    """

    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           n_targets=1, effective_rank=effective_rank, tail_strength=tail_strength, noise=noise)

    # cast to a data frame
    df = pd.DataFrame(X)
    # rename X columns
    df = rename_columns(df)
    # and add the Y
    df['y'] = y
    return df


def write_dataset(df, file_name, out_path="./"):
    """
    Writes generated dataset to file
    :param df: dataframe to write
    :param file_name: beginning of filename
    :return: none

    """
    df_train, df_testkey = train_test_split(df, test_size=.2)

    df_train.to_csv(out_path + file_name + "_train.csv", index=False)
    df_test = df_testkey.drop(['y'], axis=1)
    df_test.to_csv(out_path + file_name + "_test.csv", index=False)
    df_testkey.to_csv(out_path + file_name + "_testkey.csv", index=False)


def make_dataset(config=None):
    """
    Creates a machine learning dataset based on command line arguments passed

    :param config: a configuration dictionary, or None if called from the command line
    :return: none

    """

    if config is None:
        # called from the command line so parse configuration
        args = parse_args()
        config = load_config(args['config'])

    print('-' * 80)

    if config['type'] == 'classification':
        print('Creating Classification Dataset...')

        df = create_classification_dataset(n_samples=config['n_samples'], n_features=config['n_features'],
                                   n_informative=config['n_informative'], n_redundant=config['n_redundant'],
                                   n_repeated=config['n_duplicate'], n_clusters_per_class=config['n_clusters'],
                                   weights=config['weights'], n_classes=config['n_classes'])

    elif config['type'] == 'regression':
        print('Creating Regression  Dataset...')

        df = create_regression_dataset(n_samples=config['n_samples'], n_features=config['n_features'],
                                       n_informative=config['n_informative'], effective_rank=config['effective_rank'],
                                       tail_strength=config['tail_strength'], noise=config['noise'])

    if config['n_categorical'] > 0:
        print("Creating Categorical Features...")
        df = create_categorical_features(df, config['n_categorical'], config['label_list'])

    print("Inserting Requested Entropy...")
    # add $ or % column if requested
    if config['insert_dollar'] == "Yes":
        df = insert_special_char('$', df)
    if config['insert_percent'] == "Yes":
        df = insert_special_char('%', df)
    # insert missing values
    df = insert_missing_values(df, config['pct_missing'])

    print('Done Creating Dataset')

    print("Writing Train/Test Datasets")
    write_dataset(df, config['output'], config['out_path'])


if __name__ == "__main__":
    make_dataset()
