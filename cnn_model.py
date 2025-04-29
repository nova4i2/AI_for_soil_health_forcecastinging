import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define parameters
IMG_SIZE = (128, 128)  # Resize images
BATCH_SIZE = 32  # Number of images processed per batch

# Create an ImageDataGenerator instance
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalization and split for training/testing

# Load training images
train_images = datagen.flow_from_directory(
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\soil types",  # Set to the parent folder containing subfolders
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training"  # Training set (80%)
)

# Load validation images
val_images = datagen.flow_from_directory(
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\Soil types",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation"  # Validation set (20%)
)

# Print class indices (folder names will be mapped to labels automatically)
print("Class Indices:", train_images.class_indices)


### pre process images ###

# Define image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Image data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images (Make sure you have images organized in folders by class)
train_images = datagen.flow_from_directory("C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\Soil types",
                                           target_size=IMG_SIZE,
                                           batch_size=BATCH_SIZE,
                                           class_mode='categorical',
                                           subset="training")

val_images = datagen.flow_from_directory("C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\Soil types",
                                         target_size=IMG_SIZE,
                                         batch_size=BATCH_SIZE,
                                         class_mode='categorical',
                                         subset="validation")

### train cnn ###

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load Pre-trained ResNet50 (without the top layer)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(5, activation='softmax')  # 5 classes
])

# Compile & train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, validation_data=val_images, epochs=15)

