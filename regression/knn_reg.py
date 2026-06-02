from common_imports import *
from python_reg import *

from sklearn.neighbors import KNeighborsRegressor

model_linear_knn = KNeighborsRegressor(n_neighbors=5)
model_linear_knn.fit(X_train_reg, y_train_reg)

model_predt_knn = model_linear_knn.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, model_predt_knn))
print("MSE:", mean_squared_error(y_test_reg, model_predt_knn))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, model_predt_knn)))
print("R2 Score:", r2_score(y_test_reg, model_predt_knn))