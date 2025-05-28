from flask import Flask, render_template, request
import joblib
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load models & encoders
scaler = joblib.load('models/scaler.pkl')
encoders = joblib.load('models/encoders.pkl')
model = tf.keras.models.load_model('models/model.h5')

@app.route('/')
def home():
    venues = encoders['venue'].classes_
    teams = encoders['batting_team'].classes_
    players = encoders['striker'].classes_  # assuming striker and bowler share same pool

    return render_template('index.html', venues=venues, teams=teams, players=players)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        venue = request.form['venue']
        batting_team = request.form['batting_team']
        bowling_team = request.form['bowling_team']
        striker = request.form['striker']
        bowler = request.form['bowler']

        # Encode inputs
        input_data = np.array([
            encoders['venue'].transform([venue])[0],
            encoders['batting_team'].transform([batting_team])[0],
            encoders['bowling_team'].transform([bowling_team])[0],
            encoders['striker'].transform([striker])[0],
            encoders['bowler'].transform([bowler])[0],
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0][0]
        prediction = round(prediction)

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
