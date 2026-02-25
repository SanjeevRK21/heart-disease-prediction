from flask import Flask, render_template, request
import pickle
import numpy as np
import psycopg2
from dotenv import load_dotenv
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

def save_prediction(data):
    # Load environment variables
    load_dotenv()

    # Get database URL
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Connect
    conn = psycopg2.connect(DATABASE_URL)

    cursor = conn.cursor()
 
    query = '''INSERT INTO prediction2 (age, sex, chestpaintype, restingbp, cholesterol, 
               fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope, prediction, probability) 
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''
    
    cursor.execute(query, data)
    conn.commit()
    conn.close()

# ✅ Load the ML Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "ML_model", "best_heart_disease_model.pkl")
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/diseases")
def diseases():
    return render_template("diseases.html")

@app.route("/model_info")
def model_info():
    return render_template("model.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            input_features = np.array([float(request.form[key]) for key in request.form.keys()]).reshape(1, -1)
            
            # ✅ Ensure correct feature count before prediction
            if input_features.shape[1] != 11:
                return f"Error: Expected 11 features but received {input_features.shape[1]}."
            
            # ✅ Make Prediction
            prediction = int(model.predict(input_features)[0])  # Convert to int to avoid BLOB issue
            probability_no = float(model.predict_proba(input_features)[0][0])  # No rounding
            probability_yes = float(model.predict_proba(input_features)[0][1])  # No rounding


            data_to_save = tuple(map(float, input_features[0])) + (int(prediction), float(probability_yes))
            save_prediction(data_to_save)

            return render_template('result.html', prediction_text=prediction, 
                                   prob_no=probability_no, prob_yes=probability_yes)
        except Exception as e:
            return f"Error: {e}"

    return render_template('predict.html')


@app.route('/predict2', methods=['GET', 'POST'])
def predict2():
    if request.method == 'POST':
        try:
            # Extract input features from the form
            input_features = np.array([float(request.form[key]) for key in request.form.keys()]).reshape(1, -1)

            # Make Prediction
            prediction = int(model.predict(input_features)[0])  # Convert to int
            probability_no = float(model.predict_proba(input_features)[0][0])  # No rounding
            probability_yes = float(model.predict_proba(input_features)[0][1])  # No rounding

    
            data_to_save = tuple(input_features[0]) + (prediction, probability_yes)
            save_prediction(data_to_save)

            return render_template('result2.html', prediction_text=prediction, 
                                   prob_no=probability_no, prob_yes=probability_yes)
        except Exception as e:
            return f"Error: {e}"

    return render_template('predict2.html')

@app.route('/save_prediction2', methods=['POST'])
def save_prediction2():
    try:
        # Extract data from the form
        data = (
            request.form['age'], request.form['sex'], request.form['chestpaintype'],
            request.form['restingbp'], request.form['cholesterol'], request.form['fastingbs'],
            request.form['restingecg'], request.form['maxhr'], request.form['exerciseangina'],
            request.form['oldpeak'], request.form['st_slope'], int(request.form['prediction']),  # Convert to int
            float(request.form['probability'])  # Convert to float
        )


        conn = psycopg2.connect(
        "postgresql://postgres.yitmynulseuoekzdajpg:Ax0vvu26Apj9UlL7@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
        )
        cursor = conn.cursor()


        query = '''INSERT INTO prediction2 (age, sex, chestpaintype, restingbp, cholesterol, 
                   fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope, prediction, probability) 
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)'''

        cursor.execute(query, data)
        conn.commit()
        conn.close()

        return "Prediction saved successfully! <br><a href='/predict2'>Go Back</a>"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
