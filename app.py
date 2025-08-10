from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from disease_info import disease_data   # ✅ Import your disease info dictionary

# ✅ Create Flask app
app = Flask(__name__)

# ✅ Class names (same order as training)
class_names = [
    'Aphids', 'Army_worm', 'Bacterial_Blight', 'Curl_Virus', 'fusarium_Wilt', 'Healthy',
    'Powdery_Mildew', 'Target_Spot'
]

# ✅ Load the trained full model
MODEL_PATH = 'models/cotton_disease_efficientnet_full.keras'
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ✅ Prediction function
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # same as training

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence

# ✅ Home route (Upload page)
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filepath = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", prediction="No file selected")

        # ✅ Save file to static/uploads
        os.makedirs("static/uploads", exist_ok=True)
        filepath = os.path.join("static/uploads", file.filename)
        file.save(filepath)

        # ✅ Predict disease
        prediction, confidence = predict_disease(filepath)

        # ✅ Show prediction on the same page instead of redirecting
        return render_template("index.html",
                               prediction=prediction,
                               confidence=confidence,
                               filename=filepath)

    return render_template("index.html")


# ✅ Details page (Prescriptive care)
@app.route("/details", methods=["POST"])
def details():
    filename = request.form.get("filename")
    disease = request.form.get("disease")
    confidence = float(request.form.get("confidence"))

    # ✅ Fetch disease info
    info = disease_data.get(disease, {"symptoms": [], "favorable_conditions": [], "measures": []})

    return render_template("details.html",
                           filename=filename,
                           disease=disease,
                           confidence=confidence,
                           info=info)

# ✅ Run server
if __name__ == "__main__":
    app.run(debug=True)
