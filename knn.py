from common_imports import *

model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

predit_knn = model_knn.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predit_knn):.2f}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

# **cross validation**

score_knn = cross_val_score(model_knn, X, y, cv=5)

print(score_knn)
print(score_knn.mean())