
########################################################################
# Scores a Machine Learning Problem Dataset
# Suitable for assigning to a student for a homework assignment
#
#
########################################################################

from __future__ import print_function, absolute_import, division
import argparse
from math import sqrt

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sys


def parse_args(args):
    """
    Returns arguments passed at the command line as a dict
    :return: configuration dictionary

    """
    parser = argparse.ArgumentParser(description='Scores a ML dataset solution.')
    parser.add_argument('-p', help="Predictions File", required=True,
                        dest='pred')
    parser.add_argument('-k', help="Key File", required=True,
                        dest='key')
    return vars(parser.parse_args(args))

def read_files(y_file_name, yhat_file_name):
    """
    Opens file names and returns dataframes

    :return: y, y_hat  as a tuple of dataframes
    """
    y_hat = pd.read_csv(yhat_file_name, header=None)  # i'm expecting no header for now.  This might be a problem.
    # our test key has the features and the answer in it, we only need the answer to score
    y_df = pd.read_csv(y_file_name)
    y = y_df['y']
    return y, y_hat


def guess_problem_type(key):
    """
    Infers the problem type, using the key dataframe
    :param key: the answer dataframe
    :return: Inferred Problem Type
    """
    num_values = len(key.unique())
    if num_values == 2:
        return "binary"
    elif (num_values > 2) and (num_values < 100):  # assumptions that will burn me later probably
        return "multiclass"
    else:
        return "regression"


def score_binary_classification(y, y_hat, report=True):
    """
    Create binary classification output
    :param y: true value
    :param y_hat: class 1 probabilities
    :param report:
    :return:
    """
    y_hat_class = [1 if x >= 0.5 else 0 for x in y_hat] # convert probability to class for classification report

    report_string = "---Binary Classification Score--- \n"
    report_string += classification_report(y, y_hat_class)
    score = roc_auc_score(y, y_hat)
    report_string += "\nAUC = " + str(score)

    if report:
        print(report_string)

    return (score, report_string)


def score_multiclass_classification(y, y_hat, report=True):
    """
    Create multiclass classification score
    :param y:
    :param y_hat:
    :return:
    """
    report_string = "---Multiclass Classification Score--- \n"
    report_string += classification_report(y, y_hat)
    score = accuracy_score(y, y_hat)
    report_string += "\nAccuracy = " + str(score)

    if report:
        print(report_string)

    return(score, report_string)


def score_regression(y, y_hat, report=True):
    """
    Create regression score
    :param y:
    :param y_hat:
    :return:
    """
    r2 = r2_score(y, y_hat)
    rmse = sqrt(mean_squared_error(y, y_hat))
    mae = mean_absolute_error(y, y_hat)

    report_string = "---Regression Score--- \n"
    report_string += "R2 = " + str(r2) + "\n"
    report_string += "RMSE = " + str(rmse) + "\n"
    report_string += "MAE = " + str(mae) + "\n"

    if report:
        print(report_string)

    return(mae, report_string)


def score_dataset(y_file=None, y_hat_file=None):
    """
    1 Reads in key file and prediction file (students predictions)
    2 guesses problem type
    3 scores problem

    :return:
    """

    report_output = True
    if y_file is None and y_hat_file is None:
        # called from the command line so parse configuration
        args = parse_args(sys.argv[1:])
        y, y_hat = read_files(args['key'], args['pred'])
    else:
        y, y_hat = read_files(y_file, y_hat_file)
        report_output = False

    problem_type = guess_problem_type(y)
    print("Problem Type Detection: " + problem_type )
    print("y shape: " + str(y.shape) + " y hat shape: " + str(y_hat.shape))
    if problem_type == 'binary':
        results = score_binary_classification(y, y_hat[0], report=report_output)
    elif problem_type == 'multiclass':
        results = score_multiclass_classification(y, y_hat[0], report=report_output)
    else:
        results = score_regression(y, y_hat)

    if not report_output:
        return results


if __name__ == "__main__":
    score_dataset()

