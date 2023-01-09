from __future__ import division, print_function
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import keras.utils as image
from flask import Flask, request, render_template

app = Flask(__name__)

classes = {
    0: 'Нет малярии',
    1: 'Малярия',
}

MODEL_PATH = 'Malaria-Infected-Cells-Classification.h5'

# СЮДА ДОБАВИТЬ ПУТЬ К МОДЕЛИ
model = load_model(MODEL_PATH)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(130, 130))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def main_page():
    # Main page
    return render_template('info.html')

@app.route('/metrics', methods=['GET'])
def metrics():
    # Main page
    return render_template('carousel.html')

@app.route('/predict', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        file_path = r'uploads/' + rf'{f.filename}'
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(float(preds[0]))
        if preds[0] > 0.001:
            return 'Здоров'
        else:
            return 'Болен'
    return None


if __name__ == '__main__':
    app.run(debug=True)
