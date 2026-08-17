from flask import Flask, render_template, request
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
import numpy as np
import json
import os

from flask import Flask, render_template, request
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plant_disease_model.tflite"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "class_names.json"
)

print("Loading TFLite model...")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

print("TFLite model loaded successfully!")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Loaded {len(class_names)} classes")


def predict_disease(image):

    image = image.convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    interpreter.set_tensor(
        input_index,
        image_array
    )

    interpreter.invoke()

    predictions = interpreter.get_tensor(
        output_index
    )[0]

    predicted_index = np.argmax(predictions)

    confidence = float(
        predictions[predicted_index]
    ) * 100

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


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

                    prediction, confidence = predict_disease(
                        image
                    )

                except Exception as e:

                    print("Prediction error:", e)

                    error = "Unable to process the image."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )