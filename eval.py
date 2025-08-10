from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import numpy as np
import tensorflow as tf

# 1. Load test dataset with correct preprocessing
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    directory="dataset/test",
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

# 2. Load model
model = tf.keras.models.load_model('models/cotton_disease_efficientnetB3_full_new.keras', compile=False)

# 3. Predict
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# 4. Ground truth
y_true = test_generator.classes

# 5. Evaluation
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report")
target_names = list(test_generator.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

# 6. Additional Metrics
accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')

precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1 Score: {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall: {recall_weighted:.4f}")
print(f"Weighted F1 Score: {f1_weighted:.4f}")

