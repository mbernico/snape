import unittest
import pandas as pd
import numpy as np

from snape.score_dataset import *


class TestScoreDataset(unittest.TestCase):
    def setUp(self):
        np.random.seed(seed=42)
        y_rand = (np.random.rand(200))
        r = {'y': y_rand * 10,
             'y_hat': y_rand * 10 - y_rand
             }
        self.regression_df = pd.DataFrame(r)

        c = {'y': [1, 1, 1, 1, 0, 0, 0, 0],
             'y_hat': [1, 0.9, 0.4, 0.95, 0, 0.1, 0.6, 0.15]
             }
        self.classification_df = pd.DataFrame(c)

        m = {'y': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
             'y_hat': [0, 1, 0, 1, 1, 3, 1, 2, 2, 3, 2, 3]
             }
        self.multiclass_df = pd.DataFrame(m)

    def test_guess_problem_type(self):
        self.assertEqual(guess_problem_type(self.regression_df['y']), 'regression')
        self.assertEqual(guess_problem_type(self.classification_df['y']), 'binary')
        self.assertEqual(guess_problem_type(self.multiclass_df['y']), 'multiclass')

    def test_score_binary_classification(self):
        y = self.classification_df['y']
        y_hat = self.classification_df['y_hat']
        self.assertAlmostEquals(score_binary_classification(y,y_hat, report=False)[0], 0.9375)
        self.assertTrue("---Binary Classification Score---" in score_binary_classification(y, y_hat, report=False)[1])

    def test_score_multiclass_classification(self):
        y = self.multiclass_df['y']
        y_hat = self.multiclass_df['y_hat']
        self.assertAlmostEquals(round(score_multiclass_classification(y,y_hat, report=False)[0],2), 0.67)
        self.assertTrue("---Multiclass Classification Score---" in score_multiclass_classification(y, y_hat, report=False)[1])

    def test_score_regression(self):
        y = self.regression_df['y']
        y_hat = self.regression_df['y_hat']
        self.assertAlmostEquals(round(score_regression(y,y_hat, report=False)[0],2), 0.48)
        self.assertTrue("---Regression Score---" in score_regression(y, y_hat, report=False)[1])

if __name__ == '__main__':
    unittest.main()
