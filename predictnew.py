import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ✅ Class names (must match training order)
class_names = [
    'Aphids', 'Army_worm', 'Bacterial_Blight', 'Cotton_Boll_Rot', 'curl_virus', 'fusarium_wilt', 'Healthy', 'Leaf_Hopper_Jassids', 'Leaf_Redding', 'Leaf_Variegation', 'Powdery_Mildew', 'Target_spot', 
]

# ✅ Load the full saved model
model = tf.keras.models.load_model('models/cotton_disease_efficientnet_full_new.keras', compile=False)

# ✅ Prediction function
def predict_disease(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)  # ✅ use same preprocessing as training

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return predicted_class, confidence

if __name__ == "__main__":
    img_path = "dataset/train/Leaf Variegation/LV00015.jpg"  # 🔹 Change this to your test image path
    pred_class, conf = predict_disease(img_path)
    print(f"Predicted: {pred_class} ({conf:.2f}%)")
