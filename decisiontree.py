from common_imports import *

model_dt = tree.DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

predt_dt = model_dt.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, predt_dt):.2f}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, predt_dt))
