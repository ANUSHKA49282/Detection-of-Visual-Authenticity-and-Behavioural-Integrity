# metrics.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "microexpression_model.keras"  # Change if needed
IMG_SIZE = 75
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DATA_DIR = r"C:\Users\sunke\Downloads\archive (14)\test"  # Path to test folder

# ----------------------------
# Load model
# ----------------------------
model = load_model(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}")

# ----------------------------
# Load test data
# ----------------------------
X_test = []
y_test = []

class_map = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

for cls in CLASSES:
    cls_path = os.path.join(DATA_DIR, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=-1)
        X_test.append(img)
        y_test.append(class_map[cls])

X_test = np.array(X_test)
y_test = np.array(y_test)

# ----------------------------
# Predict
# ----------------------------
y_pred_probs = model.predict(X_test, batch_size=32, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# ----------------------------
# Metrics
# ----------------------------
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
