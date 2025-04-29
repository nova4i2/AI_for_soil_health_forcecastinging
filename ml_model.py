import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\soil_managemeent\\cnn\\soil_data_label.csv")

# Features & Target
X = df.drop(columns=["Output"])  # All features including Soil Type
y = df["Output"]  # Target is soil health (0, 1, 2)

# Normalize feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"🔍 Model Accuracy: {accuracy:.2f}\n")
print("📄 Classification Report:\n", classification_report(y_test, y_pred, target_names=["Worst", "Average", "Best"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=["Worst", "Average", "Best"], yticklabels=["Worst", "Average", "Best"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Soil Health Classification")
plt.show()
