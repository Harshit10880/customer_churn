from common_imports import *
from knn import *
from python import *


# gridcsv method implement of hyper parameter tuing using knn
model_knn = KNeighborsClassifier()

params = {
    'n_neighbors': [1, 3, 5, 7, 9]
}

params_2 = {
     'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'p': [1, 2]
}

grid_knn = GridSearchCV(
    estimator=model_knn,
    param_grid=params_2,
    cv=5
)
grid_knn.fit(X_train, y_train)


print("Best K value:", grid_knn.best_params_)
print("Best Accuracy:", grid_knn.best_score_)