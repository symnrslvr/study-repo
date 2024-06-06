import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pycaret.classification import *
data = pd.read_csv('future_new.csv')


X = data.drop('Class', axis=1)
y = data['Class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

counter = Counter(y_train)
print("Before SMOTE:", counter)

smt = SMOTE(random_state=42)
x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)

train_set = pd.concat([x_train, y_train], axis=1)
test_set = pd.concat([x_test, y_test], axis=1)

train_set.to_csv('train_latest.csv', index=False)
test_set.to_csv('test_latest.csv', index=False)

counter = Counter(y_train_sm)
print("After SMOTE:", counter)

model = ExtraTreesClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(x_train_sm, y_train_sm)

best_model = grid_search.best_estimator_
print(best_model)

y_pred_best = best_model.predict(x_test)
y_pred_best_proba = best_model.predict_proba(x_test)

print("\nBest Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred_best))

print("\nBest Model AUC Score:")
print(roc_auc_score(y_test, y_pred_best_proba, multi_class='ovr'))
