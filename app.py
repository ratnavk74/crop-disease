from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("model.h5")

IMG_SIZE = 224

classes = ["Healthy", "Leaf_Blight"]

treatments = {
    "Healthy": [
        "No disease detected",
        "Maintain proper watering",
        "Ensure balanced nutrients"
    ],
    "Leaf_Blight": [
        "Apply fungicide like Mancozeb",
        "Remove infected leaves",
        "Avoid excess moisture",
        "Improve air circulation"
    ]
}

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, [1, IMG_SIZE, IMG_SIZE, 3])
    return img

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    filepath = os.path.join("static/uploads", file.filename)
    file.save(filepath)

    img = preprocess_image(filepath)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 2)

    label = classes[index]
    solution = treatments[label]

    return render_template("result.html",
                           prediction=label,
                           confidence=confidence,
                           treatment=solution,
                           img=filepath)

if __name__ == "__main__":
    app.run(debug=True)
