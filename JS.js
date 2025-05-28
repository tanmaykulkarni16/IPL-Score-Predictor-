from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow import keras

app = Flask(__name__)c

# Load saved model and encoders
model = keras.models.load_model('models/model.h5')
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        venue = request.form['venue']
        bat_team = request.form['bat_team']
        bowl_team = request.form['bowl_team']
        batsman = request.form['batsman']
        bowler = request.form['bowler']

        # Encode inputs
        input_data = np.array([
            encoders['venue'].transform([venue])[0],
            encoders['bat_team'].transform([bat_team])[0],
            encoders['bowl_team'].transform([bowl_team])[0],
            encoders['batsman'].transform([batsman])[0],
            encoders['bowler'].transform([bowler])[0]
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0][0]
        prediction = round(prediction)

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
