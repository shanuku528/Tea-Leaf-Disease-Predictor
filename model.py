import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ⚙️ Paths
dataset_directory = "C:\\Users\\Lenovo\\OneDrive\\tea leaf\\categories"
model_filename = "model.h5"
labels_filename = "labels.txt"
confusion_matrix_image = "static/confusion_matrix.png"
os.makedirs("static", exist_ok=True)

# 📁 Categories and their paths
disease_paths = {
    "red leaf spot": os.path.join(dataset_directory, "infected/red leaf spot"),
    "white spot": os.path.join(dataset_directory, "infected/white spot"),
    "algal leaf": os.path.join(dataset_directory, "infected/algal leaf"),
    "gray blight": os.path.join(dataset_directory, "infected/gray blight"),
    "anthracnose": os.path.join(dataset_directory, "infected/anthracnose"),
    "bird eye spot": os.path.join(dataset_directory, "infected/bird eye spot"),
    "brown blight": os.path.join(dataset_directory, "infected/brown blight"),
    "helopeltis": os.path.join(dataset_directory, "infected/helopeltis"),
    "healthy": os.path.join(dataset_directory, "healthy")
}

# 🔄 Load and preprocess dataset
def preprocess_data():
    X, y = [], []
    for disease, path in disease_paths.items():
        for file in os.listdir(path):
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(path, file)
                img = load_img(img_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                X.append(img_array)
                y.append(disease)
    X = np.array(X)
    y = np.array(y)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, label_encoder

# 🧠 Build model using VGG16 as base
def build_model(num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 📊 Evaluate and show confusion matrix
def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc * 100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(confusion_matrix_image)
    plt.close()
    print(f"✅ Confusion matrix saved to {confusion_matrix_image}")

# 🚀 Train, test, evaluate, and save model
def main():
    print("📥 Loading and preprocessing data...")
    X, y_encoded, label_encoder = preprocess_data()

    X, y_encoded = shuffle(X, y_encoded, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("🧠 Building and training model...")
    model = build_model(len(label_encoder.classes_))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    print("📈 Evaluating model...")
    evaluate_model(model, X_test, y_test, label_encoder)

    print(f"💾 Saving model to {model_filename}...")
    model.save(model_filename)

    with open(labels_filename, "w") as f:
        for label in label_encoder.classes_:
            f.write(label + "\n")
    print(f"✅ Labels saved to {labels_filename}")

# 🔁 Run if executed directly
if __name__ == "__main__":
    main()
