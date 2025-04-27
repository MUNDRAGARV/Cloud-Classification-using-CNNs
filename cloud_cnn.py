import kagglehub
import os
import matplotlib.pyplot as plt
import random

# Download latest version
path = kagglehub.dataset_download("mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database")

print("Path to dataset files:", path)
path = os.path.join(path, "CCSN_v2")

import os

# List class folders
print("Available classes:")
print(os.listdir(path))

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

EPOCHS = 30

from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns


# Reuse training and validation generators
def build_model(base, name):
    base.trainable = True
    freeze_until = len(base.layers) // 3
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Model: {name}")
    model.summary()
    return model

# Xception model
xception_base = Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
xception_model = build_model(xception_base, "Xception")
history_x = xception_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
xception_model.save("xception_cloud.h5")

# MobileNetV2 (Cloud MobiNet) model
mobilenet_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
mobilenet_model = build_model(mobilenet_base, "MobileNetV2")
history_m = mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
mobilenet_model.save("mobilenet_cloud.h5")

from sklearn.metrics import classification_report
import numpy as np

def get_predictions(model, generator):
    preds = model.predict(generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes
    return y_true, y_pred

def plot_conf_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Labels
labels = list(train_gen.class_indices.keys())

# Xception Confusion Matrix
y_true_x, y_pred_x = get_predictions(xception_model, val_gen)
plot_conf_matrix(y_true_x, y_pred_x, labels, "Xception Confusion Matrix")

# MobileNetV2 Confusion Matrix
y_true_m, y_pred_m = get_predictions(mobilenet_model, val_gen)
plot_conf_matrix(y_true_m, y_pred_m, labels, "MobileNetV2 Confusion Matrix")

# Plot accuracy and loss comparison
def plot_training_curves(history_x, history_m):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axs[0].plot(history_x.history['accuracy'], label='Xception - Train')
    axs[0].plot(history_x.history['val_accuracy'], label='Xception - Val')
    axs[0].plot(history_m.history['accuracy'], label='MobileNet - Train')
    axs[0].plot(history_m.history['val_accuracy'], label='MobileNet - Val')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(history_x.history['loss'], label='Xception - Train')
    axs[1].plot(history_x.history['val_loss'], label='Xception - Val')
    axs[1].plot(history_m.history['loss'], label='MobileNet - Train')
    axs[1].plot(history_m.history['val_loss'], label='MobileNet - Val')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history_x, history_m)

from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ Add this if not defined yet
val_base_path = val_gen.directory
class_names = list(train_gen.class_indices.keys())

def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def compare_model_predictions(image_path, true_label):
    x, raw_img = load_and_preprocess(image_path)

    pred_x = xception_model.predict(x, verbose=0)[0]
    pred_m = mobilenet_model.predict(x, verbose=0)[0]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].imshow(raw_img)
    axs[0].axis('off')
    axs[0].set_title(f"Image\nTrue: {true_label}")

    axs[1].bar(class_names, pred_x, color='blue')
    axs[1].set_ylim([0, 1])
    axs[1].set_title("Xception Prediction")

    axs[2].bar(class_names, pred_m, color='green')
    axs[2].set_ylim([0, 1])
    axs[2].set_title("MobileNetV2 Prediction")

    plt.tight_layout()
    plt.show()

# 🔁 Run for a few random validation images
for _ in range(4):
    label = random.choice(os.listdir(val_base_path))
    label_dir = os.path.join(val_base_path, label)
    img_file = random.choice(os.listdir(label_dir))
    img_path = os.path.join(label_dir, img_file)
    compare_model_predictions(img_path, label)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(name, model, val_gen):
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    }

# Evaluate both models
results = []
results.append(evaluate_model("Xception", xception_model, val_gen))
results.append(evaluate_model("MobileNetV2", mobilenet_model, val_gen))

# Create a table using pandas
df_metrics = pd.DataFrame(results)
print("\n📊 Evaluation Metrics Table:\n")
print(df_metrics.to_string(index=False))

import kagglehub
import os
import matplotlib.pyplot as plt
import random

# Download latest version
path = kagglehub.dataset_download("mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database")

print("Path to dataset files:", path)
path = os.path.join(path, "CCSN_v2")

import os

# List class folders
print("Available classes:")
print(os.listdir(path))

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

EPOCHS = 30

from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns


# Reuse training and validation generators
def build_model(base, name):
    base.trainable = True
    freeze_until = len(base.layers) // 3
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Model: {name}")
    model.summary()
    return model

# Xception model
xception_base = Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
xception_model = build_model(xception_base, "Xception")
history_x = xception_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
xception_model.save("xception_cloud.h5")

# MobileNetV2 (Cloud MobiNet) model
mobilenet_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
mobilenet_model = build_model(mobilenet_base, "MobileNetV2")
history_m = mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
mobilenet_model.save("mobilenet_cloud.h5")

from sklearn.metrics import classification_report
import numpy as np

def get_predictions(model, generator):
    preds = model.predict(generator, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes
    return y_true, y_pred

def plot_conf_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.show()

# Labels
labels = list(train_gen.class_indices.keys())

# Xception Confusion Matrix
y_true_x, y_pred_x = get_predictions(xception_model, val_gen)
plot_conf_matrix(y_true_x, y_pred_x, labels, "Xception Confusion Matrix")

# MobileNetV2 Confusion Matrix
y_true_m, y_pred_m = get_predictions(mobilenet_model, val_gen)
plot_conf_matrix(y_true_m, y_pred_m, labels, "MobileNetV2 Confusion Matrix")

# Plot accuracy and loss comparison
def plot_training_curves(history_x, history_m):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axs[0].plot(history_x.history['accuracy'], label='Xception - Train')
    axs[0].plot(history_x.history['val_accuracy'], label='Xception - Val')
    axs[0].plot(history_m.history['accuracy'], label='MobileNet - Train')
    axs[0].plot(history_m.history['val_accuracy'], label='MobileNet - Val')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(history_x.history['loss'], label='Xception - Train')
    axs[1].plot(history_x.history['val_loss'], label='Xception - Val')
    axs[1].plot(history_m.history['loss'], label='MobileNet - Train')
    axs[1].plot(history_m.history['val_loss'], label='MobileNet - Val')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_training_curves(history_x, history_m)

from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ Add this if not defined yet
val_base_path = val_gen.directory
class_names = list(train_gen.class_indices.keys())

def load_and_preprocess(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0), img

def compare_model_predictions(image_path, true_label):
    x, raw_img = load_and_preprocess(image_path)

    pred_x = xception_model.predict(x, verbose=0)[0]
    pred_m = mobilenet_model.predict(x, verbose=0)[0]

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].imshow(raw_img)
    axs[0].axis('off')
    axs[0].set_title(f"Image\nTrue: {true_label}")

    axs[1].bar(class_names, pred_x, color='blue')
    axs[1].set_ylim([0, 1])
    axs[1].set_title("Xception Prediction")

    axs[2].bar(class_names, pred_m, color='green')
    axs[2].set_ylim([0, 1])
    axs[2].set_title("MobileNetV2 Prediction")

    plt.tight_layout()
    plt.show()

# 🔁 Run for a few random validation images
for _ in range(4):
    label = random.choice(os.listdir(val_base_path))
    label_dir = os.path.join(val_base_path, label)
    img_file = random.choice(os.listdir(label_dir))
    img_path = os.path.join(label_dir, img_file)
    compare_model_predictions(img_path, label)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(name, model, val_gen):
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4)
    }

# Evaluate both models
results = []
results.append(evaluate_model("Xception", xception_model, val_gen))
results.append(evaluate_model("MobileNetV2", mobilenet_model, val_gen))

# Create a table using pandas
df_metrics = pd.DataFrame(results)
print("\n📊 Evaluation Metrics Table:\n")
print(df_metrics.to_string(index=False))
# === ADDED: Imports for callbacks and evaluation ===
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

# === ADDED: Compute class weights for balancing ===
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))

# === ADDED: EarlyStopping callback ===
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# === UPDATED: Model training with class weights and early stopping ===
history_x = xception_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)

history_m = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early_stop],
    class_weight=class_weights_dict
)

# === ADDED: Evaluation and metrics reporting ===
def evaluate_model(name, model, val_gen):
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

    acc = round(accuracy_score(y_true, y_pred) * 100, 2)
    prec = round(precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    rec = round(recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)
    f1 = round(f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100, 2)

    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_true, y_pred, target_names=list(train_gen.class_indices.keys())))

    return {
        "Model": name,
        "Accuracy (%)": acc,
        "Precision (%)": prec,
        "Recall (%)": rec,
        "F1 Score (%)": f1
    }

# Evaluate both models
results = [
    evaluate_model("Xception", xception_model, val_gen),
    evaluate_model("MobileNetV2", mobilenet_model, val_gen)
]

# Display metrics table
df_metrics = pd.DataFrame(results)
print("\n📊 Evaluation Metrics (%):\n")
print(df_metrics.to_string(index=False))

# 📁 Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 📦 Step 2: Load and Build Models
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import Xception, MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import kagglehub

# 📂 Step 3: Reload Dataset and Data Generators
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = kagglehub.dataset_download("mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database")
path = os.path.join(path, "CCSN_v2")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# 🔁 Retrain Xception with Fine-Tuning
xception_base = Xception(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
xception_base.trainable = True

xception_model = models.Sequential([
    xception_base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

xception_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🔁 Retrain MobileNetV2 with Fine-Tuning
mobilenet_base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
mobilenet_base.trainable = True

mobilenet_model = models.Sequential([
    mobilenet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Compute class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights_dict = dict(enumerate(class_weights))

# Callbacks: EarlyStopping + ReduceLROnPlateau
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1)

# Train Xception
history_xception = xception_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler]
)
xception_model.save("/content/xception_cloud.h5")

# Train MobileNetV2
history_mobilenet = mobilenet_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    class_weight=class_weights_dict,
    callbacks=[early_stop, lr_scheduler]
)
mobilenet_model.save("/content/mobilenet_cloud.h5")

# 🧪 Step 4: Evaluation + Visualization
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt

# Get predictions
class_labels = list(val_gen.class_indices.keys())

def evaluate_model(model, name):
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)

    # Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(2)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

    # Classification Report Table
    print(f"\n📊 Classification Metrics for {name} (per class):\n")
    display(report_df.head(len(class_labels)))

    # Save to CSV
    report_df.to_csv(f"/content/drive/MyDrive/{name.lower()}_classification_report.csv")

    # Plot bar chart of F1 Scores
    f1_scores = report_df.loc[class_labels]['f1-score']
    f1_scores.plot(kind='bar', figsize=(10,5), title=f"F1 Score by Class - {name}", color='skyblue')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Evaluate both models
evaluate_model(xception_model, "Xception")
evaluate_model(mobilenet_model, "MobileNetV2")
