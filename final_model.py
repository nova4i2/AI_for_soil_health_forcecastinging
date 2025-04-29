# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------- STEP 1: LOAD & PREPROCESS IMAGES --------------------------- #

# Define image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Create ImageDataGenerator for training and validation (20% validation split)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training images
train_images = datagen.flow_from_directory(
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\Soil types",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training"
)

# Load validation images
val_images = datagen.flow_from_directory(
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\Soil types",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="validation"
)

# Get class labels mapping
class_labels = list(train_images.class_indices.keys())  # ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

# --------------------------- STEP 2: BUILD & TRAIN CNN MODEL --------------------------- #

# Define CNN architecture
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # 5 classes (soil types)
])

# Compile CNN model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_model.fit(train_images, validation_data=val_images, epochs=10)

# --------------------------- STEP 3: LOAD & PREPROCESS CSV DATA --------------------------- #

# Load soil properties CSV file
csv_path = "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\soil_data_label.csv"  # Update with correct path
df = pd.read_csv(csv_path)

# Check dataset structure
print(df.head())

# Define feature columns and target variable
X = df.drop(columns=["Output"])  # Keep all soil properties
  
y = df["Output"]  # Target variable (0: Worst, 1: Average, 2: Best)

# Split dataset for ML training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------- STEP 4: TRAIN ML MODEL --------------------------- #

# Train Random Forest Classifier on soil properties
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(X_train, y_train)

# Evaluate ML model
y_pred = ml_model.predict(X_test)
print("ML Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --------------------------- STEP 5: PREDICT SOIL TYPE FROM IMAGE --------------------------- #

# Define function to predict soil type using CNN
def predict_soil_type(image_path, model, class_labels):
    img = load_img(image_path, target_size=(128, 128))  # Load image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch processing
    
    prediction = model.predict(img_array)  # Predict class
    predicted_class = np.argmax(prediction)  # Get class index
    return class_labels[predicted_class]  # Return predicted soil type label

# Define soil type mapping to numerical value
soil_type_mapping = {
    "Black Soil": 0,
    "Cinder Soil": 1,
    "Laterite Soil": 2,
    "Peat Soil": 3,
    "Yellow Soil": 4
}

# --------------------------- STEP 6: MERGE CNN & ML FOR FINAL PREDICTION --------------------------- #

# Function to predict soil health using both CNN and ML models
def predict_soil_health(image_path, soil_properties_row, cnn_model, ml_model, class_labels):
    # Predict soil type from image
    soil_type = predict_soil_type(image_path, cnn_model, class_labels)
    soil_type_numeric = soil_type_mapping[soil_type]

    # 🧠 Only add soil_type if it's not already included in test data
    if len(soil_properties_row) == 12:
        input_features = np.append(soil_properties_row, soil_type_numeric).reshape(1, -1)
    else:
        input_features = soil_properties_row.reshape(1, -1)

    # Avoid feature name warnings (optional)
    input_features_df = pd.DataFrame(input_features)

    # Predict and return clean int
    predicted_health = ml_model.predict(input_features_df)
    return int(predicted_health[0])


# --------------------------- STEP 7: TEST THE HYBRID MODEL --------------------------- #

# Example test image path (update with actual test image)
test_image_paths = [
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\test_image1.jpg",
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\test_image2.jpg",
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\test_image3.jpg",
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\test_image4.jpg",
    "C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\test_image5.jpg"
]

# Example soil properties for testing (replace with real values)
test_soil_properties = np.array([
    [333, 7.7, 275, 7.54, 0.55, 0.47, 7.5, 0.36, 2.53, 0.46, 8.66, 0.18, 2],
    [213, 7.5, 338, 7.62, 0.75, 1.06, 25.4, 0.3, 0.86, 1.54, 2.89, 2.29, 1],
    [163, 9.6, 718, 7.59, 0.51, 1.11, 14.3, 0.3, 0.86, 1.57, 2.7, 2.03, 2],
    [157, 6.8, 475, 7.64, 0.58, 0.94, 26, 0.34, 0.54, 1.53, 2.65, 1.82, 3],
    [270, 9.9, 444, 7.63, 0.4, 0.86, 11.8, 0.25, 0.76, 1.69, 2.43, 2.26, 4]
])
  

# Predict soil health
for image_path, soil_props in zip(test_image_paths, test_soil_properties):
    predicted_soil_health = predict_soil_health(image_path, soil_props, cnn_model, ml_model, class_labels)
# Display result
    health_mapping = {1: "Worst quality of soil not suitable for agriculture", 2: "Average quality of soil . wont recommend it ", 3: "Best quality of soil vey suitable for agriculture"}
    print(f"Predicted Soil Health: {health_mapping[predicted_soil_health]}")
# outcomes
history = cnn_model.fit(train_images, validation_data=val_images, epochs=10)

# Plot training & validation accuracy
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Prediction distribution #

sns.countplot(x=y_pred)
plt.title("Predicted Soil Health Distribution")
plt.xlabel("Soil Health Class")
plt.ylabel("Count")
plt.xticks(ticks=[1, 2, 3], labels=["Degraded Soil", "Moderately Fertile Soil", "Highly Fertile Soil"])
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Simulated data: time in years
years = [0, 1, 2, 3, 4, 5]  # Assuming we observe over 5 years

predicted_soil_quality = {
    "Black soil": [1, 1, 2, 2, 2, 3],
    "Clinder soil": [2, 2, 2, 3, 3, 3],
    "Laterite soil": [1, 2, 2, 2, 2, 3],
    "Peat soil": [2, 2, 2, 2, 3, 3],
    "Yellow soil": [1, 1, 1, 2, 2, 2],
}

# Plotting
plt.figure(figsize=(10, 6))
for image, quality in predicted_soil_quality.items():
    plt.plot(years, quality, marker='o', label=image)

plt.title('Soil Quality Improvement Over Time')
plt.xlabel('Time (Years)')
plt.ylabel('Soil Quality')
plt.yticks([1, 2, 3], ['Degraded Soil', 'Moderately Fertile Soil', 'Highly Fertile Soil'])
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

