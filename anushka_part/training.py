# microexpression_train.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = r"C:\Users\sunke\Downloads\archive (14)\train"  # Change to your dataset path
IMG_SIZE = 75
BATCH_SIZE = 16
EPOCHS = 25

# ----------------------------
# Load dataset
# ----------------------------
print("Loading dataset...")
X = []
y = []

classes = sorted([cls for cls in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, cls))])
class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}

for cls in classes:
    cls_path = os.path.join(DATA_DIR, cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # convert to grayscale
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(class_map[cls])

X = np.array(X, dtype="float32") / 255.0
y = np.array(y)
y_cat = to_categorical(y, num_classes=len(classes))
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Dataset loaded. Classes: {classes}")
print(f"Total images: {len(X)}")

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
class_weights = {i: w for i, w in enumerate(class_weights)}
print(f"Class weights: {class_weights}")

# ----------------------------
# Build CNN model
# ----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ----------------------------
# Callbacks
# ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ----------------------------
# Train model
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------
# Evaluate
# ----------------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# ----------------------------
# Save model
# ----------------------------
model.save("microexpression_model.h5")      # legacy HDF5
model.save("microexpression_model.keras")   # modern Keras format
print("Model saved as both .h5 and .keras formats")
