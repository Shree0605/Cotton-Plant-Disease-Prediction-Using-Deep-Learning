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

# ✅ Class weights (from your data)
class_weights = {0: 0.95, 1: 0.95, 2: 0.95,3: 1.98, 4: 1.18, 
                 5: 1.18, 6: 0.95, 7: 0.85, 8: 3.84, 9: 0.85, 10: 0.95, 11: 0.95}

# ✅ Data generators
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)

val_generator = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)

# ✅ Model (Functional API to avoid multi-input bug)
input_shape = (300, 300, 3)
num_classes = 12

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)
base_model.trainable = False

inputs = tf.keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3),
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
]

# ✅ Training
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# ✅ Save the full model
model.save('models/cotton_disease_efficientnetB3_full_new.keras', include_optimizer=False)
