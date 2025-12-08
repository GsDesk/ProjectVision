import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50  # Increased epochs for better convergence

# Paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data_collection/dataset_cropped')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, '../models/face_recognition_model.h5')
INDICES_SAVE_PATH = os.path.join(BASE_DIR, '../models/class_indices.json')

def add_random_occlusion(image):
    """
    Agrega un recuadro negro aleatorio para simular oclusiones (taparse la cara).
    Se aplica aleatoriamente al 50% de las imágenes.
    """
    if np.random.random() > 0.5:
        return image
    
    h, w, _ = image.shape
    mask_size = np.random.randint(30, 60) # Tamaño del parche (30-60px)
    
    top = np.random.randint(0, h - mask_size)
    left = np.random.randint(0, w - mask_size)
    
    image[top:top+mask_size, left:left+mask_size, :] = 0 # Parche negro
    return image

def train_model():
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found.")
        return

    # Data Augmentation and Loading
    # Increased augmentation + Occlusion to fix "Phone covering face" issues
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=add_random_occlusion, # Simula objetos tapando la cara
        validation_split=0.2
    )

    # DEFINIR ORDEN FIJO DE CLASES PARA ROBUSTEZ (Evita cambios de ID)
    FIXED_CLASSES = ['Alex', 'Oscar', 'Unknown']
    
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=FIXED_CLASSES, # <--- FORZAR ORDEN
        subset='training'
    )
    
    print("Class Indices (Fixed): ", train_generator.class_indices)

    validation_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=FIXED_CLASSES, # <--- FORZAR ORDEN
        subset='validation'
    )

    if train_generator.samples == 0:
        print("No images found. Please run data collection first.")
        return

    # Calculate Class Weights to handle imbalance
    # REMOVED AUTOMATIC WEIGHTING to avoid overfitting to the tiny 'Unknown' class.
    # The 'Unknown' class has very few images, causing massive weights (e.g. 23x) which
    # forces the model to predict 'Unknown' incorrectly.
    # We will rely on the confidence threshold to reject unknown faces.
    class_weights_dict = {0: 1.0, 1: 1.0, 2: 1.0} 
    print(f"Class Weights (Manual): {class_weights_dict}")

    # Model Architecture
    # Model Architecture with MobileNetV2 (Transfer Learning & Fine Tuning)
    base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    
    # Fine Tuning: Unfreeze the top layers of the model
    base_model.trainable = True
    
    # Freeze the bottom layers (generic features) and unfreeze top layers (specific features)
    # MobileNetV2 has 155 layers total. Let's unfreeze the last 40.
    fine_tune_at = 115
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)), # Added L2 regularization
        Dropout(0.5), # Standard dropout
        Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Lower learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Save class indices (Robust Format)
    import json
    
    robust_indices = {}
    old_indices = {}

    # Try to load existing config to preserve thresholds
    if os.path.exists(INDICES_SAVE_PATH):
        try:
            with open(INDICES_SAVE_PATH, 'r') as f:
                content = json.load(f)
                # Check if it's the new format
                if content and isinstance(list(content.values())[0], dict):
                    old_indices = content
        except:
            pass

    for name, index in train_generator.class_indices.items():
        str_index = str(index)
        
        # Default threshold
        current_threshold = 0.80
        
        # Search in old config to preserve custom thresholds
        for key, val in old_indices.items():
            if isinstance(val, dict) and val.get("name") == name:
                current_threshold = val.get("threshold", 0.80)
                break
        
        # Hardcode strictness for Unknown
        if name == "Unknown": 
            current_threshold = 1.0
        
        robust_indices[str_index] = {
            "name": name,
            "threshold": current_threshold
        }

    with open(INDICES_SAVE_PATH, 'w') as f:
        json.dump(robust_indices, f, indent=4)
    print(f"Class indices saved to {INDICES_SAVE_PATH} (Robust Format)")

    model.summary()

    # Training
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weights_dict
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
