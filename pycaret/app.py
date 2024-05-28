# from flask import Flask, render_template, request, redirect

# app = Flask(__name__)

# # Ana sayfa
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Tahmin işlemi
# @app.route('/', methods=['POST','GET'])
# def predict():
#     # Gelen form verilerini al
#     Jsc = float(request.form['Jsc'])
#     Voc = float(request.form['Voc'])
#     FF = float(request.form['FF'])
#     Efficiency = float(request.form['Efficiency'])
#     Sr = float(request.form['Sr'])
#     Sp = float(request.form['Sp'])
#     Temp = float(request.form['Temp'])

#     # Burada modelinizi kullanarak tahmin yapın
#     # Tahmini sonuçları alın

#     # Sonucu gösteren bir başka HTML sayfasına yönlendir
#     return redirect('/result')

# # Tahmin sonucu sayfası
# @app.route('/result')
# def result():
#     # Burada tahmin sonuçlarını HTML olarak gösterin
#     return "Tahmin sonuçları burada görüntülenecek."

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect
import pandas as pd
from pycaret.classification import predict_model, load_model

app = Flask(__name__)

# Pycaret modelini yükle
model = load_model('pycaret/my_best_pipeline')

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Tahmin işlemi
@app.route('/', methods=['POST'])
def predict():
    # Gelen form verilerini al
    Jsc = float(request.form['Jsc'])
    Voc = float(request.form['Voc'])
    FF = float(request.form['FF'])
    Efficiency = float(request.form['Efficiency'])
    Sr = float(request.form['Sr'])
    Sp = float(request.form['Sp'])
    Temp = float(request.form['Temp'])

    # Gelen verileri bir veri çerçevesine dönüştür
    new_data = pd.DataFrame({
        'Jsc': [Jsc],
        'Voc': [Voc],
        'FF': [FF],
        'Efficiency': [Efficiency],
        'Sr': [Sr],
        'Sp': [Sp],
        'Temp': [Temp]
    })

    # Modeli kullanarak tahmin yap
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
    else:
        predicted_class = "epi on Poly-Si seed crystal"
    predicted_score = first_prediction['prediction_score']


    # Sonucu gösteren bir başka HTML sayfasına yönlendir
    return redirect('/result?predicted_class={}&predicted_score={}'.format(predicted_class, predicted_score))

# Tahmin sonucu sayfası
@app.route('/result')
def result():
    # Tahmin sonuçlarını al
    predicted_class = request.args.get('predicted_class')
    predicted_score = request.args.get('predicted_score')

    # Sonucu bir HTML şablonu içinde göstermek için render_template kullanın
    return render_template('result.html', predicted_class=predicted_class, predicted_score=predicted_score)

if __name__ == '__main__':
    app.run(debug=True)
