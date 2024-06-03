import pandas as pd

# CSV dosyasını yükleme
df = pd.read_csv('new_data_1.csv')

# 'Efficiency' sütununu seçme
efficiency_values = df['Efficiency']

# Rastgele örnek sayısını belirleme
n_samples = 20  # Örneğin 10 rastgele değer seçelim

# 'Efficiency' sütunundan rastgele örnekler seçme
random_samples = efficiency_values.sample(n=n_samples, random_state=1)

print("Rastgele Seçilen Efficiency Değerleri:")
print(random_samples)
