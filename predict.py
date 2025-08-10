# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model

# # Load model
# model = load_model('models/cotton_disease_efficientnet.h5')
# class_names = list(train_generator.class_indices.keys())  # From train.py

# def predict_image(img_path):
#     img = image.load_img(img_path, target_size=(300, 300))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.efficientnet.preprocess_input(x)
    
#     preds = model.predict(x)
#     class_idx = np.argmax(preds[0])
#     confidence = np.max(preds[0]) * 100
#     class_name = class_names[class_idx]
    
#     return class_name, confidence

# # Example usage
# if __name__ == "__main__":
#     class_name, confidence = predict_image("dataset/test/Fusarium Wilt/fus168.jpg")
#     print(f"Predicted: {class_name} ({confidence:.2f}%)")


# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # ✅ Load trained model
# MODEL_PATH = "models/cotton_disease_efficientnet.keras"  # make sure this matches the saved model name
# model = load_model(MODEL_PATH, compile=False)

# # ✅ Define class names (update later for 14 classes)
# class_names = [
#     'Aphids',
#     'Army_worm',
#     'Bacterial_Blight',
#     'Healthy',
#     'Powdery_Mildew',
#     'Target_spot',
#     'curl_virus',
#     'fussarium_wilt'
# ]

# # ✅ Function to predict disease from image
# def predict_disease(img_path):
#     img = image.load_img(img_path, target_size=(300, 300))  # same size used in training
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0

#     prediction = model.predict(img_array)
#     predicted_class = class_names[np.argmax(prediction)]
#     confidence = np.max(prediction) * 100

#     return predicted_class, confidence

# # ✅ Test prediction
# if __name__ == "__main__":
#     img_path = "dataset/test/Fusarium Wilt/fus168.jpg"  # replace with the path to your test image
#     predicted_class, confidence = predict_disease(img_path)
#     print(f"Predicted Class: {predicted_class}")
#     print(f"Confidence: {confidence:.2f}%")

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image

# # 1. Load the model (using modern TF loading)
# try:
#     # First try the recommended way
#     model = tf.keras.models.load_model('models/cotton_disease_efficientnet.keras')
# except:
#     # Fallback for older TF versions
#     model = tf.keras.models.load_model('models/cotton_disease_efficientnet.keras', compile=False)
#     model.compile(optimizer='adam',
#                  loss='sparse_categorical_crossentropy',
#                  metrics=['accuracy'])

# # 2. Class names (replace with your actual classes from train_generator)
# CLASS_NAMES = [
#     'Aphids',
#     'Army_worm',
#     'Bacterial_Blight',
#     'Healthy',
#     'Powdery_Mildew',
#     'Target_spot',
#     'curl_virus',
#     'fussarium_wilt'
# ]

# # 3. Prediction function
# def predict_image(img_path):
#     # Load and preprocess image
#     img = image.load_img(img_path, target_size=(300, 300))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = tf.keras.applications.efficientnet.preprocess_input(x)
    
#     # Make prediction
#     preds = model.predict(x)
#     class_idx = np.argmax(preds[0])
#     confidence = np.max(preds[0]) * 100
#     return CLASS_NAMES[class_idx], confidence

# # 4. Example usage
# if __name__ == "__main__":
#     disease, confidence = predict_image("dataset/test/Fusarium Wilt/fus168.jpg")
#     print(f"Predicted Disease: {disease}")
#     print(f"Confidence: {confidence:.2f}%")

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.applications import EfficientNetB3
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image

# # ✅ Rebuild the architecture exactly as in training
# num_classes = 8  # change later if you add more classes
# input_shape = (300, 300, 3)

# base_model = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.3)(x)
# output = Dense(num_classes, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=output)

# # ✅ Load only weights from your saved model
# model.load_weights('models/cotton_disease_efficientnet.h5')  # or .h5 if that's what you have

# # ✅ Class names
# class_names = [
#     'Aphids', 'Army_worm', 'Bacterial_Blight', 'Healthy',
#     'Powdery_Mildew', 'Target_spot', 'curl_virus', 'fussarium_wilt'
# ]

# # ✅ Prediction function
# def predict_disease(img_path):
#     img = image.load_img(img_path, target_size=(300, 300))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0) / 255.0
#     preds = model.predict(img_array)
#     predicted_class = class_names[np.argmax(preds)]
#     confidence = np.max(preds) * 100
#     return predicted_class, confidence

# if __name__ == "__main__":
#     img_path = "dataset/test/Fusarium Wilt/fus168.jpg"  # change to your image
#     pred_class, conf = predict_disease(img_path)
#     print(f"Predicted: {pred_class} ({conf:.2f}%)")

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.applications import EfficientNetB3
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing import image

# # ✅ Same architecture as training
# num_classes = 8
# input_shape = (300, 300, 3)

# base_model = EfficientNetB3(weights=None, include_top=False, input_shape=input_shape)
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(num_classes, activation='softmax')
# ])

# # ✅ Load weights
# model.load_weights('models/cotton_disease_efficientnet.keras')

# # ✅ Class names
# class_names = [
#     'Aphids', 'Army_worm', 'Bacterial_Blight', 'Healthy',
#     'Powdery_Mildew', 'Target_spot', 'curl_virus', 'fussarium_wilt'
# ]

# def predict_disease(img_path):
#     img = image.load_img(img_path, target_size=(300, 300))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     preds = model.predict(img_array)
#     predicted_class = class_names[np.argmax(preds)]
#     confidence = np.max(preds) * 100
#     return predicted_class, confidence

# if __name__ == "__main__":
#     img_path = "dataset/test/Fusarium Wilt/fus168.jpg"
#     pred_class, conf = predict_disease(img_path)
#     print(f"Predicted: {pred_class} ({conf:.2f}%)")

# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image

# # ✅ Load the full model directly
# model = tf.keras.models.load_model('models/cotton_disease_efficientnet_full.keras', compile=False)

# # ✅ Class names
# class_names = [
#     'Aphids', 'Army_worm', 'Bacterial_Blight', 'Healthy',
#     'Powdery_Mildew', 'Target_spot', 'curl_virus', 'fussarium_wilt'
# ]

# def predict_disease(img_path):
#     img = image.load_img(img_path, target_size=(300, 300))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     preds = model.predict(img_array)
#     predicted_class = class_names[np.argmax(preds)]
#     confidence = np.max(preds) * 100
#     return predicted_class, confidence

# if __name__ == "__main__":
#     img_path = "dataset/test/Fusarium Wilt/fus168.jpg"
#     pred_class, conf = predict_disease(img_path)
#     print(f"Predicted: {pred_class} ({conf:.2f}%)")


import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ✅ Class names (must match training order)
class_names = [
    'Aphids', 'Army_worm', 'Bacterial_Blight', 'curl_virus', 'fusarium_wilt', 'Healthy',
    'Powdery_Mildew', 'Target_spot', 
]

# ✅ Load the full saved model
model = tf.keras.models.load_model('models/cotton_disease_efficientnet_full.keras', compile=False)

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
    img_path = "dataset/test/Powdery Mildew/zoom_4.jpg"  # 🔹 Change this to your test image path
    pred_class, conf = predict_disease(img_path)
    print(f"Predicted: {pred_class} ({conf:.2f}%)")
