import pandas as pd

df = pd.read_csv('classification/temp_data.csv')


df['Voc_Jsc_Ratio'] = round(df['Voc'] / df['Jsc'], 2)
df['Rp_Ps_Ratio'] = round(df['Rp'] / df['Rs'], 4)
df['Voc2'] = round(df['Voc'] * df['Voc'], 2)

df_new = df[['Jsc','Voc','FF','Efficiency','Rs','Rp','Rate','Temp','Voc_Jsc_Ratio', 'Rp_Ps_Ratio','Voc2','Class']]


df_new.to_csv('classification/temp_f_data.csv', index=False)


