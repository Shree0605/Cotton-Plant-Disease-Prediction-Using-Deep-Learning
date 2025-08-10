import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

# Path to your minority class folder
input_folder = 'dataset/val/Leaf Variegation'
output_folder = 'dataset/val/Leaf Variegation'  # Can be same or different

# Number of augmented images to generate
num_augmented = 60  # Adjust as needed

datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load images
images = []
for fname in os.listdir(input_folder):
    path = os.path.join(input_folder, fname)
    img = cv2.imread(path)
    if img is not None:
        img = cv2.resize(img, (300, 300))
        images.append(img)

images = np.array(images)

# Generate and save new images
i = 0
for batch in datagen.flow(images, batch_size=1, save_to_dir=output_folder, save_prefix='aug', save_format='jpg'):
    i += 1
    if i >= num_augmented:
        break
