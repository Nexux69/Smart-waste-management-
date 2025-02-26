import os
import sys
import numpy as np
import tensorflow as tf
import zipfile
import gdown
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define Constants
CLASSES = ["Biodegradable", "NonBiodegradable", "No Object Found"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
IN_COLAB = 'google.colab' in sys.modules
DATASET = ""
train_gen = None
val_gen = None
model = None

# Function to Load and Extract Dataset
def loadDataForTraining(dataset):
    global DATASET
    print(f"🔄 Downloading dataset from ID: {dataset}...")

    # Define paths
    workingdir = os.getcwd()
    fileName = "waste_dataset.zip"
    extract_path = os.path.join(workingdir, "waste_dataset")

    # Download dataset using gdown
    gdown.download(id=dataset, output=fileName, quiet=False)

    # Extract dataset
    with zipfile.ZipFile(fileName, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Check if "dataset" folder exists inside extracted files
    if "dataset" in os.listdir(extract_path):
        DATASET = os.path.join(extract_path, "dataset")
    else:
        DATASET = extract_path  # Use direct extraction path

    print("✅ Dataset extracted successfully!")
    print("📂 Extracted files:", os.listdir(DATASET))  # Debugging print

# Function to Process Data for Training
def processDataForTraining():
    global train_gen, val_gen
    print("⚙️ Processing dataset for training...")

    # Ensure dataset is not empty
    if not os.path.exists(DATASET) or len(os.listdir(DATASET)) == 0:
        raise ValueError("🚨 Dataset folder is empty! Check extraction path.")

    # Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )

    # Load training data
    train_gen = datagen.flow_from_directory(
        DATASET,
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        subset="training",
        classes=CLASSES,
    )

    # Load validation data
    val_gen = datagen.flow_from_directory(
        DATASET,
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        subset="validation",
        classes=CLASSES,
    )

    print("✅ Data processing complete!")

# Function to Train the Model
def trainTheModel():
    global model
    print("🚀 Training model...")

    # Load MobileNetV2 base model
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(CLASSES), activation="softmax")(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile model
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen)

    print("✅ Model training complete!")

# Function to Evaluate Model Performance
def evaluateModel():
    print("📊 Evaluating the model...")
    plt.figure(figsize=(7, 5))
    plt.title("Loss Over Epochs", fontsize=16)
    plt.plot(model.history.history['loss'], label='Training Loss', color='blue', lw=2)
    plt.plot(model.history.history['val_loss'], label='Validation Loss', color='red', lw=2)
    plt.grid(True)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.show()
    print("✅ Model evaluation complete!")

# Function to Save Model in TFLite Format
def saveModel():
    print("💾 Saving model in TFLite format...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    # Save model file
    with open('wastemanagement.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✅ Model saved as 'wastemanagement.tflite'")

    # Auto-download in Colab
    if IN_COLAB:
        from google.colab import files
        files.download('wastemanagement.tflite')

# Running the Full Pipeline
loadDataForTraining('176U1b13disRVlVZyA3tZ4_BhV7hBvE0P')  # Dataset ID
processDataForTraining()
trainTheModel()
evaluateModel()
saveModel()
exit()
