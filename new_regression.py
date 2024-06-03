import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Veri setini oku
data = pd.read_csv("new_data_1.csv")

# Bağımsız değişkenler (X) ve bağımlı değişken (y) olarak ayır
X = data.drop(columns=['Jsc', 'Voc', 'FF', 'Efficiency', 'Sr', 'Sp'])  
y = data[['Jsc', 'Voc', 'FF', 'Efficiency', 'Sr', 'Sp']]

# Veriyi eğitim ve test setlerine böle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele Orman Regresyon modelini oluştur ve eğit
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Eğitim setinde modelin performansını değerlendir
train_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
print(f"Eğitim RMSE: {train_rmse:.2f}")

# Test setinde modelin performansını değerlendir
test_rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
print(f"Test RMSE: {test_rmse:.2f}")

# 750 derece için tahmin yap
new_data = pd.DataFrame({'Temp': [700]})
predicted_values = model.predict(new_data)
print(f"750 derece için tahmin edilen Jsc: {predicted_values[0][0]:.2f}")
print(f"750 derece için tahmin edilen Voc: {predicted_values[0][1]:.2f}")
print(f"750 derece için tahmin edilen FF: {predicted_values[0][2]:.2f}")
print(f"750 derece için tahmin edilen Efficiency: {predicted_values[0][3]:.2f}")
print(f"750 derece için tahmin edilen Sr: {predicted_values[0][4]:.2f}")
print(f"750 derece için tahmin edilen Sp: {predicted_values[0][5]:.2f}")
