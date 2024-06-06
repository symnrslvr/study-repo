# from pycaret.datasets import get_data
# data = get_data('train_set_future_new')

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

# save_model(tuned_model, 'my_best_pipeline_future_new')
# loaded_model = load_model('my_best_pipeline_future_new')
# print(loaded_model)

# evaluate_model(loaded_model)


# from pycaret.datasets import get_data
# from pycaret.classification import *

# # Yeni veri setini yükleyin
# data_new = get_data('train_set_new')

# # Otomatik ML iş akışını başlatın
# s_new = setup(data_new, target='Class', session_id=123)

# # En iyi modeli seçin
# best_new = compare_models()

# # En iyi modeli değerlendirin
# evaluate_model(best_new)

# # AUC eğrisini çizin
# plot_model(best_new, plot='auc')

# # Karışıklık matrisini çizin
# plot_model(best_new, plot='confusion_matrix')

# # Sınıflandırma raporunu çizin
# plot_model(best_new, plot='class_report')

# # Özellik önem sıralamasını görselleştirin
# plot_model(best_new, plot='feature')

# # Hassasiyet ve geri çağırma eğrisini çizin
# plot_model(best_new, plot='pr')

# # Modeli ayarlayın
# tuned_model_new = tune_model(best_new)

# # Tahminler yapın
# predictions_new = predict_model(tuned_model_new, data=data_new)
# print(predictions_new.head())

# # Ham skorlarla tahminler yapın
# predictions_raw_new = predict_model(tuned_model_new, data=data_new, raw_score=True)
# print(predictions_raw_new.head())

# # Modeli kaydedin
# save_model(tuned_model_new, 'my_best_pipeline_new_data_1')

# # Modeli yükleyin
# loaded_model_new = load_model('my_best_pipeline_new_data_1')
# print(loaded_model_new)

# # Modeli değerlendirin
# evaluate_model(loaded_model_new)


from pycaret.datasets import get_data
from pycaret.classification import *

# Veri setini yükle
data = get_data('latest')

# Otomatik ML setup'ını yap
s = setup(data, target='Class', session_id=123, fix_imbalance=True, imputation_type='iterative')

# En iyi modeli seç
best = compare_models()
print(best)

# Modeli değerlendir
evaluate_model(best)

# ROC eğrisini çiz
plot_model(best, plot='auc')

# Karmaşıklık matrisini çiz
plot_model(best, plot='confusion_matrix')

# Sınıflandırma raporunu çiz
plot_model(best, plot='class_report')

# Özellik önemini gösteren grafiği çiz
plot_model(best, plot='feature')

# Precision-recall eğrisini çiz
plot_model(best, plot='pr')

# Modeli iyileştir
tuned_model = tune_model(best)
print(tuned_model)

# Modelin tahminlerini ekrana yazdır
predictions = predict_model(tuned_model, data=data)
print(predictions.head())

# Ham skorları kullanarak modelin tahminlerini ekrana yazdır
predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
print(predictions_raw.head())

# Modeli kaydet
save_model(tuned_model, 'my_best_pipeline_future_n')

# Modeli yükle
loaded_model = load_model('my_best_pipeline_future_n')
print(loaded_model)

# Modeli değerlendir
evaluate_model(loaded_model)
