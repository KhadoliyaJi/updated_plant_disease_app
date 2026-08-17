from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# --------------------------------------------------
# Load Class Names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} classes")

# --------------------------------------------------
# Image Prediction
# --------------------------------------------------

def predict_disease(image):

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize((128, 128))

    # Convert to NumPy array
    image_array = np.array(image)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # IMPORTANT:
    # Our new model already contains the Rescaling layer,
    # so we do NOT divide by 255 here.

    # Prediction
    predictions = model.predict(image_array, verbose=0)

    # Get predicted class
    predicted_index = np.argmax(predictions[0])

    # Confidence
    confidence = float(predictions[0][predicted_index]) * 100

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please select an image."

        else:
            file = request.files["image"]

            if file.filename == "":
                error = "Please select an image."

            else:
                try:
                    image = Image.open(file)

                    prediction, confidence = predict_disease(image)

                except Exception as e:
                    print("Prediction error:", e)
                    error = "Unable to process the image."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )