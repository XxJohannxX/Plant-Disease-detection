# Step 1: Upload Kaggle API token
from google.colab import files
files.upload()  # Upload kaggle.json when prompted

# Step 2: Move it to the right directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/

# Step 3: Set permissions
!chmod 600 ~/.kaggle/kaggle.json
# Install Kaggle CLI if not installed
!pip install -q kaggle

# Download dataset
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset

# Unzip it
!unzip -q new-plant-diseases-dataset.zip -d plant_disease_data
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# Set paths
dataset_path ="/content/plant_disease_data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train"


# Parameters
img_size = 128
batch_size = 32
num_classes = 38
# Data augmentation and loading
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Get class labels
class_labels = list(train_generator.class_indices.keys())
# Custom CNN model from scratch
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train model
history=model.fit(train_generator, validation_data=val_generator, epochs=15)
# Train model and store training history
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15
)
# Plot Accuracy and Loss curves
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Step 1: Get true and predicted labels on the validation set
val_generator.reset()  # reset generator before predicting
predictions = model.predict(val_generator, verbose=1)

# Convert predicted probabilities to class indices
predicted_classes = np.argmax(predictions, axis=1)

# Get true class labels
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# Step 2: Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Step 3: Plot Confusion Matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, cmap="viridis", xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Step 4: (Optional) Print Classification Report
print("\nClassification Report:\n")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Step 1: Binarize true labels for multi-class ROC
y_true_bin = label_binarize(true_classes, classes=np.arange(num_classes))

# Step 2: Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], predictions[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Step 3: Plot ROC curves
plt.figure(figsize=(12, 10))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC–AUC Curve for Each Class')
plt.legend(loc='lower right', fontsize='small')
plt.grid(True)
plt.show()
# Class-wise confusion matrix (one vs all)
for idx, label in enumerate(class_labels):
    cm_class = confusion_matrix((true_classes == idx).astype(int),
                                (predicted_classes == idx).astype(int))

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm_class, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'Confusion Matrix for Class: {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    import pandas as pd
from sklearn.metrics import classification_report

# Generate classification report as a dictionary
report = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True)

# Convert to DataFrame for better visualization
report_df = pd.DataFrame(report).transpose()

# Display only Precision, Recall, and F1-score
metrics_table = report_df[['precision', 'recall', 'f1-score']]

# Show the table
print("📊 Precision, Recall, and F1-Score for Each Class:\n")
display(metrics_table)

# Optionally, print weighted averages at the bottom
print("\nOverall Performance (Weighted Averages):")
print(metrics_table.loc[['weighted avg']])

# Upload test image
from google.colab import files
uploaded = files.upload()

# Use the uploaded image file
import cv2
def predict_plant_disease(image_path, model, class_labels):
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[0][predicted_index]

    # Display image and prediction
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_class} ({confidence*100:.2f}%)")
    plt.axis('off')
    plt.show()

    return predicted_class, confidence

# Get uploaded image filename
image_path = list(uploaded.keys())[0]
predicted_class, confidence = predict_plant_disease(image_path, model, class_labels)
print(f"✅ Predicted: {predicted_class} ({confidence*100:.2f}%)")
