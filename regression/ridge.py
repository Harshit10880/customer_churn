from common_imports import *
from python_reg import *

model_liner_reg = LinearRegression()
model_liner_reg.fit(X_train_reg, y_train_reg)

model_predt_reg = model_liner_reg.predict(X_test_reg)

print("MAE:", mean_absolute_error(y_test_reg, model_predt_reg))
print("MSE:", mean_squared_error(y_test_reg, model_predt_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, model_predt_reg)))
print("R2 Score:", r2_score(y_test_reg, model_predt_reg))