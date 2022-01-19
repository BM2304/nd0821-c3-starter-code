import pandas as pd
import logging

logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def data_slice(features, y_test, preds):
    """
    Check metrics on specific feature
    """
    for feature in features:
        df_y_test = pd.re
