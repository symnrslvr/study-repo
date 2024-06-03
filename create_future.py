import pandas as pd

# CSV dosyasından veriyi oku
df = pd.read_csv('future_new.csv')



# Yeni özellikleri hesapla ve virgülden sonraki iki basamağı yuvarla
df['Voc_Jsc_Ratio'] = round(df['Voc'] / df['Jsc'], 2)
df['Rp_Ps_Ratio'] = round(df['FF'] / df['Sp'], 4)

# Sadece "Class" sütununu ve yeni özellikleri içeren DataFrame'i oluştur
df_new = df[['Class', 'Voc_Jsc_Ratio', 'Rp_Ps_Ratio']]

# Yeni DataFrame'i CSV dosyasına yaz
df_new.to_csv('future_data.csv', index=False)


# df['Voc_Jsc_Ratio'] = round(df['Voc'] / df['Jsc'], 2)
# df['Rp_Ps_Ratio'] = round(df['FF'] / df['Sp'], 4)

# # Yeni DataFrame'i CSV dosyasına yaz
# df.to_csv('future_new.csv', index=False)
