from pycaret.datasets import get_data
data = get_data('data2')

from pycaret.regression import *
s = setup(data, target = 'charges', session_id = 123)
best = compare_models()
evaluate_model(best)
plot_model(best, plot = 'residuals')
plot_model(best, plot = 'feature')
predict_model(best)


predictions = predict_model(best, data=data)
predictions.head()



save_model(best, 'my_best_pipeline')



loaded_model = load_model('my_best_pipeline')
print(loaded_model)

