from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

def build_model():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(3, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    return model

model = build_model()

def train_model():
    # Sadə nümunə verilənlər
    x_train = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])  # Giriş
    y_train = np.array([4, 5, 6, 7])  # Çıxış

    # LSTM üçün formata sal
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model.fit(x_train, y_train, epochs=200, verbose=0)

train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('sequence', [])
    if len(data) != 3:
        return jsonify({"error": "Daxil edilən ardıcıllığın uzunluğu 3 olmalıdır."})

    input_data = np.array(data).reshape((1, 3, 1))
    prediction = model.predict(input_data)
    return jsonify({"prediction": float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
