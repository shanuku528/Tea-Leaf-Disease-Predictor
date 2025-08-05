<<<<<<< HEAD
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask app setup
app = Flask(__name__,static_folder='static')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and labels
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess image
    img = load_img(file_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    return jsonify({'predicted_class': predicted_label, 'file_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)
=======
import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Flask app setup
app = Flask(__name__,static_folder='static')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and labels
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"
model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Load and preprocess image
    img = load_img(file_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = labels[predicted_index]

    return jsonify({'predicted_class': predicted_label, 'file_path': file_path})

if __name__ == '__main__':
    app.run(debug=True)
>>>>>>> f6d885ae88d9016db9939500fab712d66d99569d
