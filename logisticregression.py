from common_imports import *


# Initialize and Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)


print(X['Dependents'].value_counts())


y_pred = model.predict(X_test)

# Evaluate the Results
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

# cross validation

score_lotistic = cross_val_score(model, X, y, cv=5)


print(score_lotistic)
print(score_lotistic.mean())