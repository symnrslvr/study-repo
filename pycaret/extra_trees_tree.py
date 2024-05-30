import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Kendi veri kümenizi yükleyin
# Örneğin, bir CSV dosyasından veri yükleyebilirsiniz
data = pd.read_csv('train_set_combine3.csv')
X = data.drop('Class', axis=1)  # Hedef sütunu ayırın
y = data['Class']

# ExtraTreesClassifier modelini oluşturma ve eğitme
model = ExtraTreesClassifier(n_estimators=10, random_state=0)
model.fit(X, y)

# Modeldeki ilk ağacı seçme
tree = model.estimators_[0]

# Karar ağacını görselleştirme
plt.figure(figsize=(20,10))
plot_tree(tree, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True, rounded=True)
plt.show()
