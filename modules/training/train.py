import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data_collection/dataset_cropped')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '../models/face_recognition_model.h5')
INDICES_SAVE_PATH = os.path.join(BASE_DIR, '../models/class_indices.json')

def train_model():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return

    # Data Augmentation and Loading
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    print("Class Indices: ", train_generator.class_indices)

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    if train_generator.samples == 0:
        print("No images found. Please run data collection first.")
        return

    # Model Architecture
    # Model Architecture with MobileNetV2 (Transfer Learning)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False # Freeze base model

    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator.num_classes, activation='softmax') # Use softmax for multi-class
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Save class indices
    import json
    with open(INDICES_SAVE_PATH, 'w') as f:
        json.dump(train_generator.class_indices, f)
    print(f"Class indices saved to {INDICES_SAVE_PATH}")

    model.summary()

    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save Model
    models_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Visualization
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    results_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'training_results.png')
    plt.savefig(results_path)
    print(f"Training results saved to {results_path}")

if __name__ == "__main__":
    train_model()
