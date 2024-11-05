
import pandas as pd
from sklearn.model_selection import train_test_split

# CSV dosyasını okuyalım
df = pd.read_csv('classification/temp_f_data.csv')

print("Name of data:")
print(df.columns)

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

train_set.to_csv('classification/train_set_temp.csv', index=False)
test_set.to_csv('classification/test_set_temp.csv', index=False)

print("Success")
