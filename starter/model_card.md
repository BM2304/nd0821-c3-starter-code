# Model Card for random forest classifier on census data 
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Trained a Random Forest classifier using the default hyperparameters
- Model version 1.0.0
- Model date 2022/01/19

## Intended Use
- This model predict whether income exceeds $50K/yr based on census data of the public 1994 Census Database

## Training Data
- The data used in this project comes from UCI [Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income)
- The dataset contains 48842 rows and 80% of the data was used for training.

## Evaluation Data
- The remaining 20% was used for testing.

## Metrics
- Evaluation metrics include precision (0.748), recall (0.618) and fbeta (0.677)

## Ethical Considerations
- Demographic data of the 1994 Census database was used

## Caveats and Recommendations
- Hyperparameter and different models should be trained for better performance
