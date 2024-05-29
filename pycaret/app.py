from flask import Flask, render_template, request, redirect
import pandas as pd
from pycaret.classification import predict_model, load_model

app = Flask(__name__)

model = load_model('pycaret/my_best_pipeline_train')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    Jsc = float(request.form['Jsc'])
    Voc = float(request.form['Voc'])
    FF = float(request.form['FF'])
    Efficiency = float(request.form['Efficiency'])
    Sr = float(request.form['Sr'])
    Sp = float(request.form['Sp'])
    Temp = float(request.form['Temp'])

    new_data = pd.DataFrame({
        'Jsc': [Jsc],
        'Voc': [Voc],
        'FF': [FF],
        'Efficiency': [Efficiency],
        'Sr': [Sr],
        'Sp': [Sp],
        'Temp': [Temp]
    })

    predictions = predict_model(model, data=new_data)
    first_prediction = predictions.iloc[0]

    predicted_class =int( first_prediction['prediction_label'])
    if predicted_class == 0:
        predicted_class = "epi on Si(100) H-pass"
    elif predicted_class == 1:
        predicted_class = "epi on Si(100) as-grown"
    elif predicted_class == 2:
        predicted_class = "epi on Si(111) H-pass"
    elif predicted_class == 3:
        predicted_class = "epi on Si(111) as-grown"
    elif predicted_class == 4:
        predicted_class = "epi on Poly-Si seed crystal"
    elif predicted_class == 5:
        predicted_class = "epi on Si(111) interdigitated"
    elif predicted_class == 6:
        predicted_class = "epi on Si(111) mesa"
    else:
        predicted_class = "epi on Poly-Si seed crystal"
    predicted_score = first_prediction['prediction_score']


    return redirect('/result?predicted_class={}&predicted_score={}'.format(predicted_class, predicted_score))

@app.route('/result')
def result():
    predicted_class = request.args.get('predicted_class')
    predicted_score = request.args.get('predicted_score')

    return render_template('result.html', predicted_class=predicted_class, predicted_score=predicted_score)

if __name__ == '__main__':
    app.run(debug=True)
