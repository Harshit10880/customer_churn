from common_imports import *
from python_reg import *

from sklearn.linear_model import Lasso

model_linear_lasso = Lasso()
model_linear_lasso.fit(X_train_reg, y_train_reg)

model_predt_lasso = model_linear_lasso.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, model_predt_lasso))
print("MSE:", mean_squared_error(y_test_reg, model_predt_lasso))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, model_predt_lasso)))
print("R2 Score:", r2_score(y_test_reg, model_predt_lasso))