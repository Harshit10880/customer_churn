from common_imports import *    

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

predt_nb = model_nb.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, predt_nb):.2f}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))