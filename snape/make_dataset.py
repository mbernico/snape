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
import pandas as pd
import numpy as np
import argparse
import json
import random
import re

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split


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


def make_star_schema(df, out_path="./"):
    """
    
    Converts dataset to star-schema fact and dimension tables. Dimension tables are written out to CSV files,
    and the dataframe passed to the function is converted into a 'fact' table and returned as a dataframe (this
    file is NOT written out at this point because the fact table would be subject to test/train split functions,
    and dimension tables would not be). 

    :param df: Source dataframe
    :param out_path: path to write the dimension files to
    :return: dataframe with dimension table
    """
    # Internal classes (when database version is available, some of these may move outside of the scope of this function).
    def get_categorical_columns(df):
        """
        Returns a list of categorical variables from a supplied dataframe.
        
        :param df: dataframe
        :return: list of categorical columns
        """
        just_categoricals = df.select_dtypes(include=['category','object'])
        return just_categoricals.columns
    
    
    def find_dollars(text):
        """
        Identifies if a string value specifies a dollar amount. 
        
        :param text: Text string to be evaluated for whether or not it signifies a dollar amount.
        :return: Integer value: 1 for true, 0 for false. This is later used in a sum.
        """
        dollar_match = re.match(r'^\$-?\d+\.?\d+', str(text))
        if dollar_match:
            return 1
        else:
            return 0
        
        
    def find_percentages(text):
        """
        Identifies if a string value specifies a percentage. 
        
        :param text: Text string to be evaluated for whether or not it signifies a percentage.
        :return: Integer value: 1 for true, 0 for false. This is later used in a sum.
        """
        percent_search = re.search(r'^-?\d+\.?\d+[%]$', str(text))
        if percent_search:
            return 1
        else:
            return 0
    
    
    def is_special_char(list_object):
        """
        Identifies whether or not a list object consists entirely of special characters added to the dataset. 
        
        :param list_object: List-like object; evaluated for whether or not all values within it are special character fields.
        :return: Boolean, True or False
        """
        if list_object.dtype != 'O':
            return False
        else:
            percent_sum = sum(list_object.apply(find_percentages))
            dollars_sum = sum(list_object.apply(find_dollars))

            if (percent_sum/list_object.count() == 1) or (dollars_sum/list_object.count() == 1):
                return True
            else:
                return False
    
    
    # Get the categorical columns
    cols = get_categorical_columns(df)
    assert len(cols) > 0, "No categorical variables exist in this dataset; star schema cannot be developed."
    
    # Iterate through the categorical columns
    for cat_column in cols:
        
        # Determine if the list includes requested entropy or not (NOTE: Decided not to make dimension 
        # tables before this command so dimension keys CAN'T be selected for entropy)
        if is_special_char(df[cat_column]) != True:
            
            # Turn the value counts into a dataframe
            vals = pd.DataFrame(df[cat_column].value_counts())
            
            # Reset the index to add index as the key
            vals.reset_index(inplace=True) # Puts the field names into the dataframe
            vals.reset_index(inplace=True) # Puts the index numbers in as integers
            
            # Name the column with the same name as the column 'value_count'
            vals.rename(index=str, columns={'level_0':'primary_key', 'index':'item',
                                            cat_column:'value_count'}, inplace=True)
            
            # Make a df out of just the value and the mapping
            val_df = vals[['primary_key','item']]
            
            # Make a dimension df by appending a NaN placeholder
            val_df.item.cat.add_categories('Not specified', inplace=True)
            val_df = val_df.append({'primary_key': -1, 'item': 'Not specified'}, ignore_index=True)
            
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
            #df[cat_column + '_key'] = df[cat_column + '_key'].apply(int)
            
            # Drop cat column from the dataframe
            df.drop(cat_column, axis=1, inplace=True)
            
    # Now, reset the dataframe's index and rename the index column as 'primary_key'
    df.reset_index(inplace=True)
    df_cols = df.columns
    df_cols = df_cols.delete(0)
    df_cols = df_cols.insert(0, 'primary_key')
    df.columns=df_cols

    # Return the main dataframe as a 'fact' table, which will then be split into test/train splits, since dimension tables are immune to this
    return df.copy()


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
    
    # Convert dataset to star schema if requested
    if config['star_schema'] == "Yes":
        # Check the number of categorical variables
        if config['n_categorical'] > 0:
            df = make_star_schema(df, config['out_path'])
        else:
            print("No categorical variables added. Dataset cannot be transformed into a star schema. Dataset will be generated as a single-table dataset...")

    print("Writing Train/Test Datasets")
    write_dataset(df, config['output'], config['out_path'])


if __name__ == "__main__":
    make_dataset()
