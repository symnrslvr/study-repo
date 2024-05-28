# load sample dataset
from pycaret.datasets import get_data
data = get_data('data2')

from pycaret.classification import *
s = setup(data, target='Class', session_id=123)

# Model Karşılaştırma
best = compare_models()
print(best)

# Modeli Değerlendirme
evaluate_model(best)
plot_model(best, plot='auc')
plot_model(best, plot='confusion_matrix')
plot_model(best, plot='class_report')
plot_model(best, plot='feature')
plot_model(best, plot='pr')

# Hiperparametre Ayarı
tuned_model = tune_model(best)
print(tuned_model)

# Tahmin Yapma
predictions = predict_model(tuned_model, data=data)
print(predictions.head())

predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
print(predictions_raw.head())

# Modeli Kaydetme ve Yükleme
save_model(tuned_model, 'my_best_pipeline')
loaded_model = load_model('my_best_pipeline')
print(loaded_model)

# Yüklenen Modeli Değerlendirme
evaluate_model(loaded_model)
