from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
#from model import model
import pickle
import numpy as np

app = Flask(__name__)

# Load pre-trained machine learning model
loaded_model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user input from the form
        input_data = request.form.to_dict()
        
        # Preprocess the input data for your model (if needed)
        features = [
            "cap-shape", "cap-surface", "cap-color", "gill-spacing", "gill-size",
            "gill-color", "stalk-surface-above-ring", "stalk-color-above-ring",
            "ring-type", "bruises", "odor", "spore-print-color", "population", "habitat"
        ]
        # Continue preprocessing here...
        
        prediction = loaded_model.predict(preprocessed_data)

        return render_template("result.html", prediction=prediction[0])

    else:
        return "Something went wrong. Please try again."

if __name__ == "__main__":
    app.run(debug=True)
