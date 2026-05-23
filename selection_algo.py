from common_imports import *

# to check every model using loop

models_check = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': tree.DecisionTreeClassifier(),
    'SVM': SVC()
}

result_check = []

from sklearn.metrics import f1_score, accuracy_score

for name_ch, model_ch in models_check.items():
  model_ch.fit(X_train, y_train)
  check_predt = model_ch.predict(X_test)

  # Calculate the Accuracy and F1 Score
  accuracy = accuracy_score(y_test, check_predt)
  f1 = f1_score(y_test, check_predt)

  # Append the results (rounded to 4 decimal places) into the list
  result_check.append({
      'Model': name_ch,
      'Accuracy': round(accuracy, 4),
      'F1 Score': round(f1, 4)
  })

result_check