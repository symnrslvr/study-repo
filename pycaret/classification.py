from pycaret.datasets import get_data
data = get_data('data')

from pycaret.classification import *
s = setup(data, target='Class', session_id=123)

best = compare_models()
print(best)

evaluate_model(best)
plot_model(best, plot='auc')
plot_model(best, plot='confusion_matrix')
plot_model(best, plot='class_report')
plot_model(best, plot='feature')
plot_model(best, plot='pr')

tuned_model = tune_model(best)
print(tuned_model)

predictions = predict_model(tuned_model, data=data)
print(predictions.head())

predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
print(predictions_raw.head())

save_model(tuned_model, 'my_best_pipeline_new')
loaded_model = load_model('my_best_pipeline_new')
print(loaded_model)

evaluate_model(loaded_model)
