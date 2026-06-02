from common_imports import *
from python_reg import *
from sklearn.linear_model import Ridge

mode_linear_ridge = Ridge()
mode_linear_ridge.fit(X_train_reg, y_train_reg)

model_predt_ridge = mode_linear_ridge.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, model_predt_ridge))
print("MSE:", mean_squared_error(y_test_reg, model_predt_ridge))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, model_predt_ridge)))
print("R2 Score:", r2_score(y_test_reg, model_predt_ridge))