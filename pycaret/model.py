# import pandas as pd
# from pycaret.classification import setup, load_model, predict_model, evaluate_model

# new_data = pd.read_csv('test_set_future_c.csv') 

# data = pd.read_csv('train_set_future_c.csv') 

# s = setup(data, target='Class', session_id=123)

# loaded_model = load_model('my_best_pipeline_future_c')

# new_predictions = predict_model(loaded_model, data=new_data)

# print(new_predictions)

import pandas as pd
from pycaret.classification import setup, load_model, predict_model, evaluate_model, compare_models, plot_model, tune_model, save_model

# Test setini yükle
new_data = pd.read_csv('test_set_future_c.csv') 

# Eğitim setini yükle
data = pd.read_csv('train_set_future_c.csv') 

# Pycaret setup fonksiyonunu kullanarak otomatik ML setup'ını yap
s = setup(data, target='Class', session_id=123)

# En iyi modeli seç ve ekrana yazdır
best = compare_models()
print(best)

# Modeli değerlendir
evaluate_model(best)

# AUC grafiğini çizdir
plot_model(best, plot='auc')

# Karmaşıklık matrisi grafiğini çizdir
plot_model(best, plot='confusion_matrix')

# Sınıf raporu grafiğini çizdir
plot_model(best, plot='class_report')

# Özelliklerin önem derecesi grafiğini çizdir
plot_model(best, plot='feature')

# Precision-Recall grafiğini çizdir
plot_model(best, plot='pr')

# Modeli iyileştir
tuned_model = tune_model(best)
print(tuned_model)

# Eğitim seti üzerinde modelin tahminlerini yap ve ekrana yazdır
predictions = predict_model(tuned_model, data=data)
print(predictions.head())

# Ham skorları kullanarak eğitim seti üzerinde modelin tahminlerini yap ve ekrana yazdır
predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
print(predictions_raw.head())

# Modeli kaydet
save_model(tuned_model, 'my_best_pipeline_future_c')

# Modeli yükle
loaded_model = load_model('my_best_pipeline_future_c')
print(loaded_model)

# Modeli değerlendir
evaluate_model(loaded_model)
