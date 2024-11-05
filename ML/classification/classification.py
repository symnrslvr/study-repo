
from pycaret.datasets import get_data
from pycaret.classification import *


data = get_data('classification/temp_f_data')

s = setup(data, target='Class', session_id=123, fix_imbalance=True, imputation_type='iterative')

best = compare_models()
# print(best)

evaluate_model(best)

plot_model(best, plot='auc')

plot_model(best, plot='confusion_matrix')

plot_model(best, plot='class_report')

plot_model(best, plot='feature')

plot_model(best, plot='pr')

tuned_model = tune_model(best)
print(tuned_model)

predictions = predict_model(tuned_model, data=data)
print(predictions)
predictions.to_csv('classification/predictions_temp.csv', index=False)

predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
print(predictions_raw)
predictions_raw.to_csv('classification/predictions_raw_temp.csv', index=False)

save_model(tuned_model, 'classification/my_best_pipeline_temp_data')

loaded_model = load_model('classification/my_best_pipeline_temp_data')
print(loaded_model)

evaluate_model(loaded_model)
