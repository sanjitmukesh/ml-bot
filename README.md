# Spamify Model
**Deep Learning Spam Classifier**

This is a deep learning pipeline that classifies SMS messages as spam or legitimate.
It preprocesses raw text data, converts it into numerical sequences through vectorization, and trains a neural network to accurately identify spam.

Used in the deployed web app [Spamify](https://github.com/sanjitmukesh/spamify-app).

## Features
- Preprocesses and vectorizes text messages for model training
- Trains a deep neural network using TensorFlow/Keras
- Achieves 97% validation accuracy on unseen SMS data

## Model Overview
- **Dataset:** [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  
- **Layers:** TextVectorization → Embedding → GlobalAveragePooling → Dense (ReLU) → Dense (Sigmoid)  
- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  

## Technologies
**Languages:** Python  
**Frameworks/Libraries:** TensorFlow, Keras, NumPy, Pandas, Scikit-learn
