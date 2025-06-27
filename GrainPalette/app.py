from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
model = load_model("model/rice_model.h5")

# Load class labels
with open("model/rice_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Home page
@app.route("/")
def index():
    return render_template("index.html")

# Upload form page
@app.route("/details")
def details():
    return render_template("details.html")

# Prediction handler
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files["image"]
    filepath = os.path.join("static", "uploaded_image.jpg")
    file.save(filepath)

    # Image preprocessing
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    result = labels[np.argmax(prediction)]

    return render_template("results.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)