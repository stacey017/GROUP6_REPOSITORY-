
# Data Scientist Modeling Strategy

## Dataset
UCI Heart Disease Dataset, 303 observations and 13 input features.

## Target
The original `num` target is converted to binary classification:
- 0 = No Disease
- 1 = Disease

## Data Split
An 80/20 stratified train/test split is used.

## Preprocessing
Numeric features:
- median imputation
- StandardScaler

Categorical features:
- most-frequent imputation
- OneHotEncoder with unknown-category handling

Preprocessing is placed inside sklearn Pipelines so it is fitted
within each training fold rather than before cross-validation.

## Models
- Logistic Regression: baseline
- Decision Tree: nonlinear candidate
- Random Forest: ensemble candidate
- SVM: kernel-based candidate

## Cross-Validation
5-fold StratifiedKFold with shuffle=True and random_state=42.

The primary selection metric is F1.

## Hyperparameter Tuning
GridSearchCV is applied to Decision Tree, Random Forest and SVM.

All five metrics are calculated during the same grid-search CV:
accuracy, precision, recall, F1 and ROC-AUC.

The best hyperparameters are selected using CV F1. The baseline and tuned candidates are compared using the same CV-based selection metric.

## Model Selection
The baseline and tuned candidates are compared using CV F1.
The final model is selected before the test set is evaluated.

## Bias-Variance Analysis
Training and cross-validation performance are compared.
A large train-CV gap is treated as a possible indicator of
higher variance/overfitting.

## Final Selected Model
SVM

## Final Test Results
Accuracy: 0.8525
Precision: 0.8065
Recall: 0.8929
F1: 0.8475
ROC-AUC: 0.9502

## Limitation
The dataset is small and the final test set contains only
61 observations. Therefore, these results describe
performance on this experiment and should not be interpreted
as clinical validation or deployment-level evidence.
