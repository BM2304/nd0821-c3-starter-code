# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.model import train_model
from ml.data import process_data
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
data_path = os.path.join(os.getcwd(), 'data', 'cleaned_census.csv')
data = pd.read_csv(data_path)
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

# Train and save a model.
model = train_model(X_train, y_train)
logging.info('SUCCESS: Training model')

pkl_model = 'trained_model.pkl'
with open(pkl_model, 'wb') as f:
    pickle.dump(model, f)

logging.info('SUCCESS: Saving model')
