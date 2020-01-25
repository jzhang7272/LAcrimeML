import sys
sys.path.append('/usr/local/lib/python3.7/site-packages')
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets
import lightgbm as lgb
import eli5
from eli5.sklearn import PermutationImportance
from lightgbm import LGBMClassifier
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from hyperopt.pyll.base import scope

df = pd.read_csv('/Users/jzhang/PycharmProjects/CrimeML/venv/clean.csv', delimiter=',')
df.dropna(inplace=True)

train, test = train_test_split(df, test_size=0.2)

train.drop(columns='Unnamed: 0', inplace=True)
index = test['Unnamed: 0']
test.drop(columns='Unnamed: 0', inplace=True)
test.pop('Category')

# Permutation Importance
le1 = LabelEncoder()
y = le1.fit_transform(train.pop('Category'))

print(test.dtypes)
print(train.dtypes)

train_X, val_X, train_y, val_y = train_test_split(train, y)
print(val_X.columns.tolist())

model = LGBMClassifier(objective='multiclass', num_class=28).fit(train_X, train_y)

perm = PermutationImportance(model).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())

# Initial Model
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/jzhang/PycharmProjects/CrimeML/venv/clean.csv', delimiter=',')
df.dropna(inplace=True)

train, test = train_test_split(df, test_size=0.2)

train.drop(columns='Unnamed: 0', inplace=True)
index = test['Unnamed: 0']
test.drop(columns='Unnamed: 0', inplace=True)

test_y = test.pop('Category')

le1 = LabelEncoder()
X = train.drop(columns=['Category'])
y = le1.fit_transform(train['Category'])

train_data = lgb.Dataset(X, label=y, free_raw_data=False)

params = {
    'objective': 'multiclass',
    'num_class': 27
}

cv_results = lgb.cv(params, train_data, metrics='multi_logloss', early_stopping_rounds=10)
print('Best score: ', min(cv_results['multi_logloss-mean']))
num_boost_round = np.argmin(cv_results['multi_logloss-mean'])
print('Best epoch: ', num_boost_round)

# Training
bst = lgb.train(params, train_data, num_boost_round=num_boost_round)

# Predictions
predictions = bst.predict(test)

print (log_loss(test_y, predictions))


# Bayesian Optimization
N_FOLDS = 10

# Create the dataset
train_set = lgb.Dataset(train, label = y)

def objective(params, n_folds = N_FOLDS):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
    
    # Perform n_fold cross validation with hyperparameters
    # Use early stopping and evalute based on ROC AUC
    cv_results = lgb.cv(params, train_set, nfold = n_folds, num_boost_round = 100, early_stopping_rounds = 10, 
                        metrics = 'auc', seed = 50)
  
    # Extract the best score
    best_score = max(cv_results['auc-mean'])
    
    # Loss must be minimized
    loss = 1 - best_score
    
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

from hyperopt.pyll.base import scope

# Define the search space
spaces = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type',['gbdt','goss']),
    'num_leaves': scope.int(hp.quniform('num_leaves', 20, 150, 1)),
    'max_delta_step': hp.uniform('max_delta_step', 0.0, 6.0),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 20000, 300000, 20000)),
    'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 500, 5)),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

# Algorithm
tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()

MAX_EVALS = 500

# Optimize
best = fmin(fn = objective, space = spaces, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
print(best)


# Final Model
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/jzhang/PycharmProjects/CrimeML/venv/clean.csv', delimiter=',')
df.dropna(inplace=True)

train, test = train_test_split(df, test_size=0.2)

train.drop(columns='Unnamed: 0', inplace=True)
index = test['Unnamed: 0']
test.drop(columns='Unnamed: 0', inplace=True)

test_y = test.pop('Category')

# Naive Prediction
naive_vals = train.groupby('Category').count().iloc[:,0]/train.shape[0]
n_rows = test.shape[0]

submission = pd.DataFrame(
    np.repeat(np.array(naive_vals), n_rows).reshape(27, n_rows).transpose(),
    columns=naive_vals.index)

submission.to_csv('/Users/jzhang/PycharmProjects/CrimeML/venv/naive.csv', index_label='Id')

print (log_loss(test_y, np.repeat(np.array(naive_vals), n_rows).reshape(27, n_rows).transpose()))

# Building Final Model
le1 = LabelEncoder()
X = train.drop(columns=['Category'])
y = le1.fit_transform(train['Category'])

train_data = lgb.Dataset(X, label=y)

params = {
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'num_class': 27,
    'max_delta_step': 3.228508082950985,
    'min_data_in_leaf': 440,
    'learning_rate': 0.016483058783105322,
    'num_leaves': 28
}

# Training
best = lgb.train(params, train_data, num_boost_round=100)

# Predictions
predictions = best.predict(test)

print (log_loss(test_y, predictions))

# Submitting the results
submission = pd.DataFrame(predictions, columns=le1.inverse_transform(np.linspace(0, 26, 27, dtype='int16')), index=test.index)
submission.to_csv('/Users/jzhang/PycharmProjects/CrimeML/venv/lgbm_final.csv', index_label='Id')




