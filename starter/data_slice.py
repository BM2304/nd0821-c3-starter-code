from ml.data import process_data
from ml.model import compute_model_metrics, inference
import pandas as pd
import numpy as np


def slice_metrics(cat_features, test, model, encoder, lb):
    """
    Check metrics on specific feature and writes metrics for all unique values in file
    """
    slice_metrics = []
    for cat in cat_features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]
            X_test, y_test, _, _ = process_data(
                df_temp, categorical_features=cat_features, label="salary", training=False,
                encoder=encoder, lb=lb)

            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)
            slice_metrics.append(f"Category: {cat} - Value: {cls} - Precision: {precision:.3f} "
                                 f"Recall: {recall:.3f}, Fbeta: {fbeta:.3f}")

    # Write metrics to file
    with open('./model/slice_output.txt', 'w') as f:
        for slice_metric in slice_metrics:
            f.write(slice_metric + '\n')
