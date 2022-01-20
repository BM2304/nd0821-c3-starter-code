# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.model import compute_model_metrics, inference, train_model
from ml.data import process_data
from data_slice import slice_metrics
import pandas as pd
import logging
import pickle
import os

# Logging
logging.basicConfig(
    filename='./results.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# Load in the cleaned data
data = pd.read_csv('./starter/data/cleaned_census.csv')
logging.info('SUCCESS: Loaded cleaned census.csv data')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)
logging.info('SUCCESS: Split data in training and test data')

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)
logging.info('SUCCESS: Preprocessing data')

# Train model and report predictions.
model = train_model(X_train, y_train)
logging.info('SUCCESS: Training model')

# prediction
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
logging.info(
    f"Precision: {precision:.3f}, Recall: {recall:.3f}, Fbeta: {fbeta:.3f}")

# test on slice data
slice_metrics(cat_features, test, model, encoder, lb)
logging.info(f'SUCCESS: Saved slice metrics on {cat_features}')

# save model, encoder and label_binarizer
pkl_model = './starter/model/trained_model.pkl'
pkl_encoder = './starter/model/binarizer.pkl'
pkl_lb = './starter/model/lb.pkl'

with open(pkl_model, 'wb') as f:
    pickle.dump(model, f)

with open(pkl_encoder, 'wb') as f:
    pickle.dump(encoder, f)

with open(pkl_lb, 'wb') as f:
    pickle.dump(lb, f)

logging.info('SUCCESS: Saving model, encoder and label binarizer in ml folder')
