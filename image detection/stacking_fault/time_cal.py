import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("stacking_fault/max_lengths.csv")

buyutme_hizi_nm_dakika = 200

# Kalınlık (örneğin, 2 mikron)
kalinhik_mikron = 2  

# Büyütme hızını mikron/dak cinsine çevir
buyutme_hizi_mikron_dakika = buyutme_hizi_nm_dakika / 1000  

# Her üçgen için defektin oluşma anını hesapla
defect_formation_times = []
for index, row in df.iterrows():
    max_length = row["Max_Length"]
    defekt_olusma_anı_dakika = (kalinhik_mikron - max_length) / buyutme_hizi_mikron_dakika
    defect_formation_times.append(round(defekt_olusma_anı_dakika * 60, 2))  # Dakika cinsinden değeri saniyeye çevirerek ekleyelim

# Yeni sütunu DataFrame'e ekle
df["Defect_Formation_Time_Seconds"] = defect_formation_times

# Yeni DataFrame'i CSV dosyasına kaydet
df.to_csv("stacking_fault/rate_defect_formation_times.csv", index=False)