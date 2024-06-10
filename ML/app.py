from flask import Flask, render_template, request, redirect
import pandas as pd
from pycaret.classification import predict_model, load_model

app = Flask(__name__)

model = load_model('ML/classification/my_best_pipeline_feature_data')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    # Formdaki alan adlarını ve bu alanlardaki değerleri bir dizi içinde toplama
    form_fields = ['Jsc', 'Voc', 'FF', 'Efficiency', 'Rs', 'Rp', 'Rate', 'Temp']
    form_data = [float(request.form[field]) for field in form_fields]

    # Form verilerini kullanarak pandas DataFrame oluşturma
    new_data = pd.DataFrame([form_data], columns=form_fields)

    # Modeli kullanarak tahmin yapma
    predictions = predict_model(model, data=new_data)
    first_prediction = predictions.iloc[0]

    predicted_class = int(first_prediction['Label'])
    class_mapping = {
        0: "epi on Si(100) H-pass",
        1: "epi on Si(100) as-grown",
        2: "epi on Si(111) H-pass",
        3: "epi on Si(111) as-grown",
        4: "epi on Poly-Si seed layer on glass"
    }
    predicted_class = class_mapping.get(predicted_class, "Unknown")
    predicted_score = first_prediction['Score']

   
    return redirect('/result?predicted_class={}&predicted_score={}'.format(predicted_class, predicted_score))

@app.route('/result')
def result():
    # Tahmin edilen sınıfı ve tahmin skorunu alma
    predicted_class = request.args.get('predicted_class')
    predicted_score = request.args.get('predicted_score')

    # Sonucu gösterme
    return render_template('result.html', predicted_class=predicted_class, predicted_score=predicted_score)

if __name__ == '__main__':
    app.run(debug=True)
