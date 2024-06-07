import pandas as pd
from pycaret.classification import setup, load_model, predict_model, evaluate_model, compare_models, plot_model, tune_model, save_model


new_data = pd.read_csv('ML/classification/test_set_wo_feature.csv') 

data = pd.read_csv('ML/classification/train_set_wo_feature.csv') 


s = setup(data, target='Class', session_id=123, fix_imbalance=True, imputation_type='iterative')


best = compare_models()
print(best)

evaluate_model(best)

# plot_model(best, plot='auc')

# plot_model(best, plot='confusion_matrix')

# plot_model(best, plot='class_report')

# plot_model(best, plot='feature')

# plot_model(best, plot='pr')

tuned_model = tune_model(best)
print(tuned_model)

predictions = predict_model(tuned_model, data=data)
print(predictions)

predictions_raw = predict_model(tuned_model, data=data, raw_score=True)
# print(predictions_raw.head())

save_model(tuned_model, 'my_best_pipeline_l')

loaded_model = load_model('my_best_pipeline_l')
print(loaded_model)

evaluate_model(loaded_model)
