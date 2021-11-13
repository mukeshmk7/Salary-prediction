from flask import Flask, render_template, request
import joblib

model = joblib.load('model.pkl')
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    score = float(request.form.get("TestScore"))
    experience = int(request.form.get("Experience"))
    int_score = int(request.form.get("InterviewScore"))
    output = round(model.predict([[experience, score, int_score]])[0], 2)
    return render_template("result.html", output=output)


app.run(debug=True)
