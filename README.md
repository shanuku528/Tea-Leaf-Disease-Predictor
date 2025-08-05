# Tea Leaf Disease Predictor 🌿

## 🔍 Overview

Tea is one of the most widely consumed beverages in the world, but tea plants are highly susceptible to various leaf diseases. Early and accurate disease detection is essential to protect yield and quality. This project implements a **Convolutional Neural Network (CNN)**-based system to automatically detect and classify tea leaf diseases from images.

The system includes:

* **Training Script** (`model.py`) to build and train the CNN model
* **Flask Web Application** (`app.py`) for real-time predictions
* **Web Interface** using HTML, CSS, and JavaScript

## 🌐 Features

* Classifies tea leaves into multiple categories:

  * Healthy
  * Red Leaf Spot
  * White Spot
  * Algal Leaf
  * Gray Blight
  * Anthracnose
  * Bird Eye Spot
  * Brown Blight
  * Helopeltis
* Web-based interface for image upload & instant predictions
* Confusion matrix visualization after training
* JSON-based API responses for easy integration

## 📂 Project Structure

```
Tea-Leaf-Disease-Predictor/
├── app.py                  # Flask application for predictions
├── model.py                # CNN model training and evaluation
├── model/                  # Model files (model.json, model.h5*)
├── static/                 # CSS, images, static assets
│   ├── style.css
│   └── background.jpg
├── templates/              # HTML templates
│   ├── index.html
│   └── predict.html
├── labels.txt              # Class labels generated during training
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files/folders
└── README.md               # Project documentation
```

> \*Note: `model.h5` is larger than 100MB and cannot be directly pushed to GitHub without Git LFS. See instructions below.

## 📊 Model Training (`model.py`)

* Uses **VGG16** as the base model with transfer learning
* Trains on categorized tea leaf images
* Saves:

  * **`model.h5`** (trained weights)
  * **`labels.txt`** (mapping from class indices to labels)
* Generates confusion matrix (`static/confusion_matrix.png`)

## 📒 Prediction Web App (`app.py`)

* Loads trained model & labels
* Accepts uploaded images
* Preprocesses image (resizes to 128x128, normalizes)
* Runs prediction and returns result as JSON
* Displays result in browser with image preview

## 🔗 Model Download

Due to GitHub's 100MB file limit, the trained model file (`model.h5`) is hosted externally.

**Download here:** [Google Drive Link](https://drive.google.com/drive/u/0/folders/1ih0tib-Oz0uSWMleBS81qZ90B_l9wMyV)

Place the downloaded `model.h5` inside the project root folder before running the app.

## 📁 Dataset

Dataset used for training contains categorized images of healthy and diseased tea leaves.

Source: [Kaggle Tea Leaf Disease Dataset](https://www.kaggle.com/datasets/shashwatwork/identifying-disease-in-tea-leafs)

## 🐍 Python Version

* Python 3.9+ recommended

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/shanuku528/Tea-Leaf-Disease-Predictor.git
cd Tea-Leaf-Disease-Predictor
```

2. **Create virtual environment** (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download model file** (see link above) and place in project folder.

## 🚶 Usage

Run the Flask app:

```bash
python app.py
```

Visit **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

Upload a tea leaf image and get instant prediction results.

## 🛋️ .gitignore

```gitignore
# Python cache
__pycache__/
*.pyc

# Virtual environment
venv/

# IDE configs
.idea/

# Large model files
*.h5
*.hdf5

# Temporary uploads
temp/
uploads/

# Dataset (optional)
dataset/
```

## 📄 requirements.txt (sample)

```txt
Flask
numpy
matplotlib
seaborn
tensorflow
scikit-learn
```

## 💡 Future Enhancements

* Deploy on cloud (Heroku / AWS / Azure)
* Add mobile-friendly UI
* Include more diseases
* Model optimization for faster inference

## 🛆 Contributors

* **[Shanu Ku](https://github.com/shanuku528)** – Developer & Researcher
* **[Mohammeed Ramees Ummer](https://github.com/Ramees903757)** – Developer & Researcher

---

**📓 License:** MIT License
