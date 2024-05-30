from sklearn.model_selection import cross_val_score

# Model
model = RandomForestClassifier(random_state=42)

# Cross-validation yap
cv_scores = cross_val_score(model, x_test, y_test, cv=5)

# Sonuçları yazdır
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
