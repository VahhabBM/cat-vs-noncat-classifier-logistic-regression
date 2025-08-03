# 🐱 cat-vs-noncat-classifier-logistic-regression

🧠 **Logistic Regression from Scratch (Cat vs Non-Cat Classifier using NumPy)**  
A simple binary image classifier built from scratch using only NumPy. Trained to detect whether an image contains a cat or not. Dataset provided by deeplearning.ai.

---

## 📌 Overview

This project implements a simple logistic regression model to classify images as **cat** (1) or **non-cat** (0), using only NumPy.

### It includes:
- 🧾 Labeled dataset of cat/non-cat images
- ⚙️ Vectorized implementation with NumPy
- 🔁 Sigmoid activation & gradient descent
- 📊 Accuracy evaluation on training & test data
- 📉 Cost visualization during training

---

## 🧠 What the Code Does

1. Loads and preprocesses image data (flatten + normalize)
2. Initializes weights and bias to zero
3. Trains the model using logistic regression + gradient descent
4. Plots cost reduction over iterations
5. Predicts labels and prints accuracy on both training and test sets

---

## 🚀 Run the Code

1. 📦 Make sure you have the dataset files:
   - `train_catvnoncat.h5`
   - `test_catvnoncat.h5`
   - and the helper file: `lr_utils.py`
   > Place them in the same folder as your script.

2. 🛠 Install required dependencies:
```bash
pip install numpy matplotlib h5py pillow scipy
