import pandas as pd
import numpy as np
from scipy import stats

# Veri kümesini oluşturma
data = {
    'Efficiency':[4.39,3.17,3.53,6.01,1.49,4.40,6.21,6.24,4.89,3.92,3.40,4.38,3.67,4.79,2.34,4.78,4.06,6.01,4.97,4.42]
    #[0.39,0.94,1.04,1.27,5.70,0.92,0.96,0.93,0.84,1.01,1.13,0.98,1.34,0.88,0.94,5.91,0.94,1.27,0.48,0.89]
    #[0.78,2.19,4.09,4.15,0.02,0.02,2.50,1.04,2.75,2.86,2.75,4.13,0.03,4.30,0.16,3.06,2.19,1.05,1.11,0.84] 
    #[3.54,4.29,4.16,3.95,1.41,1.04,3.31,3.40,3.26,3.08,3.21,4.18,2.23,4.21,2.20,4.27,2.81,3.23,2.37,1.84]
    #[1.33,1.43,0.23,3.95,5.65,3.77,5.20,5.70,4.92,5.81,6.34,5.26,2.67,5.46,6.20,3.58,0.21,0.45,4.18,5.52]
   # 'Efficiency': [4.17,4.06,3.83,4.45,4.97,6.44,4.24,6.12,4.85,6.00,3.20,5.23,6.40,6.01,6.01,6.55,3.81,6.45,4.39,5.41]
}

# DataFrame'e dönüştürme
df = pd.DataFrame(data)

# 'Efficiency' sütununu seçme
efficiency = df['Efficiency']

# Örneklem ortalamasını hesaplama
mean = efficiency.mean()
print("Örneklem Ortalaması:", mean)

# Örneklem standart sapmasını hesaplama
std_dev = efficiency.std(ddof=1)
print("Örneklem Standart Sapması:", std_dev)

# Örneklem büyüklüğünü hesaplama
n = len(efficiency)
print("Örneklem Büyüklüğü:", n)

# Güven düzeyi ve kritik z değeri (burada %95 güven düzeyi için z ≈ 1.96)
confidence_level = 0.95
z = stats.norm.ppf((1 + confidence_level) / 2)
print("Z Değeri:", z)

# Standart hatayı hesaplama
standard_error = std_dev / np.sqrt(n)
print("Standart Hata:", standard_error)

# Güven aralığını hesaplama
margin_of_error = z * standard_error
confidence_interval = (mean - margin_of_error, mean + margin_of_error)
print("Güven Aralığı:", confidence_interval)
