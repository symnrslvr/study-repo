# Gerekli kütüphaneleri import edelim
import pandas as pd
from sklearn.model_selection import train_test_split

# CSV dosyasını okuyalım
df = pd.read_csv('latest.csv')

# Sütun adlarını kontrol edelim
print("Veri çerçevesinin sütun adları:")
print(df.columns)

# Özellikler (features) ve hedef değişkeni (label) ayıralım
# 'target' sütunu hedef değişkeni temsil etmektedir, bunu kendi veri setinize göre ayarlayabilirsiniz
X = df.drop('Class', axis=1)
y = df['Class']

# Veriyi eğitim ve test setlerine bölelim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Eğitim ve test setlerini birleştirerek ayrı DataFrame'ler oluşturalım
train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

# Eğitim ve test setlerini CSV dosyaları olarak kaydedelim
train_set.to_csv('train_set_l.csv', index=False)
test_set.to_csv('test_set_l.csv', index=False)

print("Eğitim ve test setleri başarıyla kaydedildi.")
