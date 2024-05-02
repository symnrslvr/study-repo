import pandas as pd

# If your dataset is a local CSV file
data = pd.read_csv('/home/seyma/study-repo/pycaret/data2.csv')

# If your dataset is available at a URL
# data = pd.read_csv('http://url_to_your_data/data2.csv')

print(data.head())  # This will display the first few rows of your dataset