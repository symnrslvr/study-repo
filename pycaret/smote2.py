import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pycaret.classification import *
# Veri kümesini yükleyin ve hazırlayın
# Burada örnek bir veri seti oluşturuyorum. Kendi veri setinizi buraya yüklemelisiniz.
data = pd.read_csv('future_new.csv')

# Özellikleri ve hedef değişkeni ayırın
X = data.drop('Class', axis=1)
y = data['Class']

# Eğitim ve test veri setlerine bölün
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE uygulamadan önce sınıf dağılımını kontrol edin
counter = Counter(y_train)
print("Before SMOTE:", counter)

# SMOTE uygulayın
smt = SMOTE(random_state=42)
x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)

train_set = pd.concat([x_train, y_train], axis=1)
test_set = pd.concat([x_test, y_test], axis=1)

train_set.to_csv('train_latest.csv', index=False)
test_set.to_csv('test_latest.csv', index=False)

# SMOTE uygulandıktan sonra sınıf dağılımını kontrol edin
counter = Counter(y_train_sm)
print("After SMOTE:", counter)

# Modeli eğitin
model = ExtraTreesClassifier(random_state=42)

# Hiperparametre Optimizasyonu
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Cross-validation stratejisi
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
grid_search.fit(x_train_sm, y_train_sm)

best_model = grid_search.best_estimator_
print(best_model)
# En iyi model ile yeniden değerlendirme
y_pred_best = best_model.predict(x_test)
y_pred_best_proba = best_model.predict_proba(x_test)

print("\nBest Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

print("\nBest Model Classification Report:")
print(classification_report(y_test, y_pred_best))

print("\nBest Model AUC Score:")
print(roc_auc_score(y_test, y_pred_best_proba, multi_class='ovr'))


# from sklearn.model_selection import cross_val_score
# from pycaret.datasets import get_data
# data = train_set

# from pycaret.classification import *
# s = setup(data, target='Class', session_id=123)

# best = compare_models()
# print(best)

# evaluate_model(best)
# plot_model(best, plot='auc')
# plot_model(best, plot='confusion_matrix')
# plot_model(best, plot='class_report')
# plot_model(best, plot='feature')
# plot_model(best, plot='pr')

# tuned_model = tune_model(best)
# print(tuned_model)

# predictions = predict_model(tuned_model, data=data)
# print(predictions.head())

# predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
# print(predictions_raw.head())

# save_model(tuned_model, 'my_best_pipeline_train_combine3')
# loaded_model = load_model('my_best_pipeline_train_combine3')
# print(loaded_model)

# evaluate_model(loaded_model)
# # Model
# model = RandomForestClassifier(random_state=42)

# # Cross-validation yap
# cv_scores = cross_val_score(model, x_test, y_test, cv=5)

# # Sonuçları yazdır
# print("Cross-Validation Scores:", cv_scores)
# print("Mean CV Score:", cv_scores.mean())


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE
# from collections import Counter
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Veri kümesini yükleyin ve hazırlayın
# data = pd.read_csv('future_new.csv')

# # Özellikleri ve hedef değişkeni ayırın
# X = data.drop('Class', axis=1)
# y = data['Class']

# # Eğitim ve test veri setlerine bölün
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # SMOTE uygulamadan önce sınıf dağılımını kontrol edin
# counter = Counter(y_train)
# print("Before SMOTE:", counter)

# # SMOTE uygulayın
# smt = SMOTE(k_neighbors=4, random_state=42)
# x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)

# # SMOTE uygulandıktan sonra sınıf dağılımını kontrol edin
# counter = Counter(y_train_sm)
# print("After SMOTE:", counter)

# # Modeli eğitin
# model = ExtraTreesClassifier(random_state=42)
# model.fit(x_train_sm, y_train_sm)

# # Test seti üzerinde modeli değerlendirin
# y_pred = model.predict(x_test)
# y_pred_proba = model.predict_proba(x_test)

# # Model performansını değerlendirin
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# print("\nAUC Score:")
# print(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))

# # Grafikler
# # Confusion Matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Classification Report
# plt.figure(figsize=(8, 6))
# report = classification_report(y_test, y_pred, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# report_df.drop(['accuracy', 'macro avg', 'weighted avg'], inplace=True)
# report_df['precision'].plot(kind='bar', color='skyblue', label='Precision')
# report_df['recall'].plot(kind='bar', color='orange', label='Recall')
# report_df['f1-score'].plot(kind='bar', color='lightgreen', label='F1-Score')
# plt.title('Classification Report')
# plt.xlabel('Class')
# plt.ylabel('Score')
# plt.legend()
# plt.show()

# # ROC Curve
# from sklearn.metrics import plot_roc_curve
# plot_roc_curve(model, x_test, y_test)
# plt.title('ROC Curve')
# plt.show()
