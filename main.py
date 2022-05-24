from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("static/assets/Cleaned_data.csv")
pipe = pickle.load(open("static/assets/RidgeModel.pkl", "rb"))

@app.route("/")
def index():
    locations = sorted(data["location"].unique())
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():
    location = request.form.get("location") 

    try: 
        bhk = float(request.form.get("bhk"))
    except Exception as e:
        bhk = 2.0

    try:
        bath = float(request.form.get("bath"))
    except Exception as e:
        bath = 2.0

    try:
        sqft = float(request.form.get("total_sqft"))
    except Exception as e:
        sqft = 1000.0  


    print(location, bhk, bath, sqft)
    input = pd.DataFrame([[location, sqft, bath, bhk]], columns=["location", "total_sqft", "bath", "bhk"])
    
    prediction = pipe.predict(input)[0] * 1e5
    
    return str(np.round(prediction, 2))


if __name__ == "__main__":
    app.run(debug=True)