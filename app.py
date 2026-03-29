from flask import Flask, render_template, request
import pickle
import os
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
encoders = pickle.load(open(os.path.join(BASE_DIR, "encoders.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()

    df = pd.DataFrame([data])

    numeric_cols = [
        'age', 'cgpa', 'internships', 'projects',
        'programming_languages', 'certifications',
        'experience_years', 'hackathons', 'research_papers',
        'skills_score', 'soft_skills_score', 'resume_length_words'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df).max()

    if prediction == 1:
        result = f"Candidate is likely to be HIRED 🎉 (Confidence: {round(prob*100,2)}%)"
    else:
        result = f"Candidate is NOT likely to be hired ❌ (Confidence: {round(prob*100,2)}%)"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)