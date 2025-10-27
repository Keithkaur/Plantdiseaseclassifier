# 🌿 Plant Disease Classifier Using CNN

A Convolutional Neural Network (CNN)-based deep learning model for **automatic detection and classification of plant diseases** using leaf images.  
This project leverages the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) to train a robust classifier capable of identifying multiple crop diseases with high accuracy.

---

## 📘 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [How to Run](#how-to-run)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## 🧩 Overview
Early detection of plant diseases is crucial for ensuring food security and sustainable agriculture.  
This project builds a **CNN-based classifier** that identifies whether a plant leaf is healthy or diseased, and if diseased, detects the type of disease.

The model is trained on thousands of labeled leaf images, learning visual patterns indicative of different plant diseases.

---

## 🌱 Dataset
**Dataset:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  

**Description:**  
- Contains over **50,000 images** of healthy and diseased leaves.  
- Covers **14 crop species** and **30+ plant diseases**.  
- Well-labeled and balanced dataset suitable for deep learning image classification tasks.

**Preprocessing Steps:**
- Image resizing and normalization  
- Data augmentation (rotation, zoom, shear, flip)  
- Train/validation/test split  

---

## 🛠️ Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **Pillow (PIL)**
- **OS / JSON / ZipFile**

---

## 📁 Project Structure
CNN_plantDisease_classifier/
│
├── CNN_plantDisease_classifier.ipynb # Main Jupyter notebook
├── dataset/ # Contains PlantVillage images
├── models/ # Saved trained model files
├── results/ # Training results and plots
└── README.md # Project documentation


---

## 🧠 Model Architecture
The CNN model is built using **TensorFlow Keras**, with multiple convolutional and pooling layers to extract hierarchical image features.

**Typical Layers:**
- Convolutional Layers with ReLU activation  
- MaxPooling Layers  
- Dropout Layers for regularization  
- Fully Connected Dense Layers  
- Softmax Output Layer for multi-class classification  

**Loss Function:** `CategoricalCrossentropy`  
**Optimizer:** `Adam`  
**Metrics:** `Accuracy`  

---

## 📈 Training & Evaluation
- **Data Augmentation:** Applied using `ImageDataGenerator`  
- **Batch Size:** 32  
- **Epochs:** 25–50 (tunable)  
- **Validation Split:** 20%  

During training, model performance is visualized via accuracy and loss curves.

Example:
```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
