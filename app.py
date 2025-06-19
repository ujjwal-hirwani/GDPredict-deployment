from flask import Flask, render_template, request, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///feedback.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class Feedback(db.Model):
    feedback_id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(50), nullable=False)
    message = db.Column(db.String(1000), nullable=False)

    def __repr__(self):
        return f"{self.name} - {self.email} - {self.message}"
    

# Load model
model = pickle.load(open('model.pkl', 'rb'))
le_region = pickle.load(open('label_encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.form

            region = data['region']
            region_encoded = le_region.transform([region])[0]

            input_values = [
                region_encoded,
                int(data['Population']),
                int(data['Area']),
                float(data['Density']),
                float(data['Coastline']),
                float(data['NetMigration']),
                float(data['Literacy']),
                float(data['Phones']),
                float(data['Arable']),
                float(data['Crops']),
                float(data['Climate']),
                float(data['Birthrate']),
                float(data['Deathrate']),
                float(data['Agriculture']),
                float(data['Industry']),
                float(data['Service'])
            ]

            # Define column names exactly as in your dataset
            columns = [
                'Region',
                'Population',
                'Area (sq. mi.)',
                'Pop. Density (per sq. mi.)',
                'Coastline (coast/area ratio)',
                'Net migration',
                'Literacy (%)',
                'Phones (per 1000)',
                'Arable (%)',
                'Crops (%)',
                'Climate',
                'Birthrate',
                'Deathrate',
                'Agriculture',
                'Industry',
                'Service'
            ]

            # Create DataFrame
            input_df = pd.DataFrame([input_values], columns=columns)
            # Scale input
            input_scaled = scaler.transform(input_df)

            # Predict
            prediction = model.predict(input_scaled)[0]

            prediction_text = f"${prediction:.2f}"
            return redirect(url_for('result', prediction_text = prediction_text))  # Redirect with #output anchor
        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        feedback = Feedback(name = name, email = email, message = message)
        db.session.add(feedback)
        db.session.commit()
    return render_template('index.html', message = "")

@app.route('/result/<string:prediction_text>', methods=['GET', 'POST'])
def result(prediction_text):
    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=False)
