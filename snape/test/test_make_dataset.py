import unittest
import pandas as pd

from snape.make_dataset import *


class TestMakeDataset(unittest.TestCase):
    def test_create_classification_dataset(self):
        df = create_classification_dataset(n_samples=100, n_features=10, n_informative=3, n_redundant=0,
                                           n_repeated=0, n_clusters_per_class=2, weights=[0.5, 0.5], n_classes=2)

        self.assertEqual(df.shape[0], 100, "Sample Size Doesn't Match")
        self.assertEqual(df.shape[1], 11, "Feature Count")
        self.assertEqual(df['y'].value_counts().shape[0], 2, "Expected Shape of Classes Do Not Match")

    def test_create_regression_dataset(self):
        df = create_regression_dataset(n_samples=100, n_features=10, n_informative=3, effective_rank=1,
                                       tail_strength=0.5, noise=0.0)

        self.assertEqual(df.shape[0], 100, "Sample Size Doesn't Match")
        self.assertEqual(df.shape[1], 11, "Feature Count")

    def test_create_categorical_features(self):
        df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
        cat_df = create_categorical_features(df, 2, [['a', 'b'], ['red', 'blue']])
        self.assertEqual(cat_df.dtypes.value_counts()['category'], 2)  # there should be 2 category variables

    def test_insert_special_char(self):
        df = pd.DataFrame(np.random.randn(100, 1), columns=list('A'))
        df_spec = insert_special_char("$", df)
        self.assertTrue(df_spec['A'].str.contains('$').all())
        df_spec = insert_special_char("%", df)
        self.assertTrue(df_spec['A'].str.contains('$').all())

    def test_insert_missing_values(self):
        df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
        df_result = insert_missing_values(df, 1)
        self.assertTrue(df_result.isnull().any().any())
        df_result = insert_missing_values(df, 0)
        self.assertFalse(df_result.isnull().any().any())


if __name__ == '__main__':
    unittest.main()
