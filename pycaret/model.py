import pandas as pd
from pycaret.classification import setup, load_model, predict_model, evaluate_model

new_data = pd.read_csv('test_set_combine3.csv') 

data = pd.read_csv('train_set_combine3.csv') 

s = setup(data, target='Class', session_id=123)

loaded_model = load_model('my_best_pipeline_train_combine3')

new_predictions = predict_model(loaded_model, data=new_data)

print(new_predictions)

