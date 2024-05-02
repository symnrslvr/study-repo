# load sample dataset
from pycaret.datasets import get_data
data = get_data('data2')

from pycaret.classification import *
s = setup(data, target='Class', session_id=123)


best = compare_models()  # SVM, Decision Tree ve Random Forest modellerini dışlar



print(best)

evaluate_model(best)
plot_model(best, plot = 'auc')

plot_model(best, plot = 'confusion_matrix')

plot_model(best, plot = 'class_report')


predict_model(best)


predictions = predict_model(best, data=data)
predictions.head()

predictions = predict_model(best, data=data, raw_score=True)
predictions.head()

save_model(best, 'my_best_pipeline')

loaded_model = load_model('my_best_pipeline')
print(loaded_model)

