import unittest
import pandas as pd
from src.data_agent import load_data, clean_data, synchronize_data

class TestDataAgent(unittest.TestCase):

    def test_load_data(self):
        df = load_data("CDX.NA.IG.5Y")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    def test_clean_data(self):
        df = pd.DataFrame({'Close': [1, 2, None, 4]})
        df_cleaned = clean_data(df)
        self.assertFalse(df_cleaned['Close'].isnull().any())
        self.assertEqual(df_cleaned['Close'][2], 2.0)

    def test_synchronize_data(self):
        d1 = {'Close': [1, 2, 3]}
        idx1 = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])
        df1 = pd.DataFrame(data=d1, index=idx1)

        d2 = {'Close': [4, 5, 6]}
        idx2 = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04'])
        df2 = pd.DataFrame(data=d2, index=idx2)

        df1_sync, df2_sync = synchronize_data(df1, df2)

        self.assertEqual(len(df1_sync), 2)
        self.assertEqual(len(df2_sync), 2)
        self.assertTrue((df1_sync.index == df2_sync.index).all())

if __name__ == '__main__':
    unittest.main()
