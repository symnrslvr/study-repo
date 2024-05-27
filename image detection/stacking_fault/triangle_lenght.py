import pandas as pd

df = pd.read_csv('lenght_size.csv')

max_lengths = df.groupby('trianle_name')['lenght'].max()

result_df = pd.DataFrame({
    'Triangle': max_lengths.index,
    'Max_Length': max_lengths.values
})

result_df.to_csv('stacking_facult/max_lengths.csv', index=False)
