import pandas as pd
from pycaret.classification import setup, load_model, predict_model, evaluate_model

# Yeni veri setini CSV dosyasından yükleyin
new_data = pd.read_csv('new_csv.csv')  # Burada kendi dosya yolunuzu belirtin

# Eğitim verileriyle setup() işlemini tekrarlayın
data = pd.read_csv('data2.csv')  # Eğitim veri setinizin yolunu belirtin

s = setup(data, target='Class', session_id=123)

# Modeli yükleyin
loaded_model = load_model('my_best_pipeline')

# Yüklenen model ile yeni verilerde tahmin yapın
new_predictions = predict_model(loaded_model, data=new_data)

# Tahminlerin başını görüntüleyin
print(new_predictions.head())

# Modelin performansını yeni verilerde değerlendirin
# evaluate_model(loaded_model)
