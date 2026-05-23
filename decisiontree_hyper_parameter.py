from common_imports import *
from decisiontree import *
from python import *


model_dt_new = tree.DecisionTreeClassifier()
params_dt_1 = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2, 3, 4, 5],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'splitter': ['best', 'random']
}

params_dt_2 = {
        'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30],
    'ccp_alpha': [0.0, 0.01, 0.02]
}

grid_dt_1 = GridSearchCV(
    estimator=model_dt_new,
    param_grid=params_dt_1,
    cv=5
)

grid_dt_1.fit(X_train, y_train)

print("Best Parameters:")
print(grid_dt_1.best_params_)

print("\nBest Accuracy:")
print(grid_dt_1.best_score_)