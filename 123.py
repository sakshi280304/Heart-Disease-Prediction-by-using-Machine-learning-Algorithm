# Import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load models
all_models = pickle.load(open('models.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route('/aboutUs', methods=['GET'])
def about_us():
    return render_template('aboutUs.html')

@app.route('/api', methods=['GET', 'POST'])
def predict():
    name = request.form['name']
    email = request.form['email']
    age = request.form['age']
    fgender = request.form['gender']
    cp = request.form['cp']
    trestbps = request.form['trestbps']
    chol = request.form['chol']
    fbs = request.form['fbs']
    restecg = request.form['restecg']
    thalach = request.form['thalach']
    exang = request.form['exang']
    oldpeak = request.form['oldpeak']
    slope = request.form['slope']
    ca = request.form['ca']
    thal = request.form['thal']

    # Set default values for missing data
    if trestbps == '':
        trestbps = 95
    if chol == '':
        chol = 150
    if thalach == '':
        thalach = 72
    if oldpeak == '':
        oldpeak = 2

    # Convert categorical values to numeric
    gender = 1 if fgender == "Male" else 0
    thal_mapping = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
    thal = thal_mapping.get(thal, 0)
    restecg_mapping = {"Normal": 0, "STT Abnormality": 1, "LVH": 2}
    restecg = restecg_mapping.get(restecg, 0)
    exang = 1 if exang == "Yes" else 0

    # Convert input values to integers
    age = int(age)
    cp = int(cp)
    trestbps = int(trestbps)
    chol = int(chol)
    fbs = int(fbs)
    thalach = int(thalach)
    oldpeak = float(oldpeak)
    slope = int(slope)
    ca = int(ca)

    # Features for prediction
    features = [age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Prediction logic
    disease_types = {
        "Coronary Artery Disease": 0,
        "Myocardial Infarction (Heart Attack)": 0,
        "Arrhythmia": 0
    }
    predictions = {}
    avg = 0

    for model in all_models:
        res = model.predict([features])
        if res[0] == 1:
            disease_types["Coronary Artery Disease"] += 1
        elif res[0] == 2:
            disease_types["Myocardial Infarction (Heart Attack)"] += 1
        elif res[0] == 3:
            disease_types["Arrhythmia"] += 1
        predictions[model] = res[0]
        avg += res[0]

    # Calculate disease probability
    total_models = len(all_models)
    for disease in disease_types:
        disease_types[disease] = round((disease_types[disease] / total_models) * 100, 2)

    # Determine final result
    most_likely_disease = max(disease_types, key=disease_types.get)
    final_result = f"{most_likely_disease} detected with {disease_types[most_likely_disease]}% probability."

    # Prepare response data
    input_data = {
        "Age": age,
        "Gender": fgender,
        "Chest Pain Types": cp,
        "Resting Blood Pressure (mm/Hg)": trestbps,
        "Cholesterol Level": chol,
        "Fasting Blood Sugar > 120 mg/dL": fbs,
        "Resting Electrocardiographic Results": restecg,
        "Max Heart Rate Achieved": thalach,
        "Exercise Induced Angina": exang,
        "ST Depression Induced by Exercise": oldpeak,
        "Slope of Peak Exercise ST Segment": slope,
        "Major Vessels Colored by Fluoroscopy": ca,
        "Thalassemia Type": thal
    }

    personal_info = [name, email]
    responses = [input_data, predictions, personal_info, final_result, disease_types]

    return render_template("result.html", result=responses)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
