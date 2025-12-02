import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = 'modules/data_collection/dataset'

if os.path.exists(DATA_DIR):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        DATA_DIR,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical'
    )
    print("Class Indices:", generator.class_indices)
else:
    print("Dataset directory not found.")
