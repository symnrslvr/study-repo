import pandas as pd

df = pd.read_csv('stacking_fault/rate_l_s.csv')

max_lengths = df.groupby('trianle_name')['lenght'].max()

result_df = pd.DataFrame({
    'Triangle': max_lengths.index,
    'Max_Length': max_lengths.values
})

result_df.to_csv('stacking_fault/max_lengths_rate.csv', index=False)
