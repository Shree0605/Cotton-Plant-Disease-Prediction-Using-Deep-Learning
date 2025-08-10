from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ✅ Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)
# ✅ Data generators
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)

# Get class indices mapping (e.g., {'Aphids': 0, 'Army_worm': 1, ...})
class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
num_classes = len(class_names)

# Get labels for each image in the training generator
labels = train_generator.classes

# Compute class weights
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

# Create a dictionary like {0: weight0, 1: weight1, ..., 11: weight11}
class_weights = {i: weight for i, weight in enumerate(class_weights_array)}

# Optional: Print weights for debugging
print("Computed class weights:", class_weights)
