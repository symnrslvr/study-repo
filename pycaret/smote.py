from pycaret.classification import *
from pycaret.datasets import get_data
from imblearn.over_sampling import SMOTE
from collections import Counter

# Veri setini yükleyin
data = get_data('train_set_new')

# Otomatik ML iş akışını başlatın
s = setup(data, target='Class', session_id=123)

# SMOTE uygulayın
# Veri setinizin dengesizliğini kontrol edin
counter = Counter(data['Class'])
print("Before SMOTE:", counter)

# SMOTE uygulayın
smt = SMOTE()
X_sm, y_sm = smt.fit_resample(data.drop('Class', axis=1), data['Class'])

# Veri setinizin dengesizliğini kontrol edin
counter = Counter(y_sm)
print("After SMOTE:", counter)

# Eğitim veri setini ve hedef değişkeni yeniden tanımlayın
data_sm = X_sm.copy()
data_sm['Class'] = y_sm

# Otomatik ML iş akışını başlatın
s = setup(data_sm, target='Class', session_id=123)

# En iyi modeli seçin
best_model = compare_models()

# En iyi modeli ayarlayın
tuned_model = tune_model(best_model)

# Modeli kaydedin
save_model(tuned_model, 'my_best_pipeline_with_smote')

# Modeli yükleyin
loaded_model = load_model('my_best_pipeline_with_smote')

# Modeli değerlendirin
evaluate_model(loaded_model)

# AUC eğrisini çizin
plot_model(tuned_model, plot='auc')

# Karışıklık matrisini göster
plot_model(tuned_model, plot='confusion_matrix')

# Sınıflandırma raporunu göster
plot_model(tuned_model, plot='class_report')

# Özellik önem sıralamasını göster
plot_model(tuned_model, plot='feature')
