# Introduction to Computer Vision & Deep Learning 🧠👁️

> **From "Hello World" to Real-World Image Classification:** A journey through the fundamentals of Neural Networks using TensorFlow and Keras.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Introduction_to_TensorFlow_Wrap_Up.ipynb)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg) ![Keras](https://img.shields.io/badge/Keras-High%20Level%20API-red.svg) ![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 📖 About This Project

This repository documents my progression from a complete beginner in Machine Learning to building a Convolutional Neural Network (CNN) capable of classifying real-world images.

Instead of just copying code, this notebook represents a narrative of **learning by doing**. It starts with a single neuron and evolves into a multi-layer deep learning architecture, tackling problems like **Overfitting** and **Data Scarcity** along the way.

---

## 🚀 The Learning Journey

The project is divided into **4 Modules**, each introducing a new level of complexity:

### **Module 1: The "Hello World" of Deep Learning**
* **The Goal:** Teach a computer the relationship $Y = 2X - 1$.
* **The Concept:** I learned how a **single neuron** (Dense layer) uses **Stochastic Gradient Descent (SGD)** and **Mean Squared Error (MSE)** to "learn" a math formula without being explicitly programmed with rules.

### **Module 2: The Limitation of Dense Networks**
* **The Goal:** Classify clothing items using the **Fashion MNIST** dataset.
* **The Challenge:** Using standard Dense layers required "flattening" the images, destroying the 2D spatial relationships.
* **The Result:** The model struggled to recognize objects if they were shifted or complex. This proved the need for a better architecture.

### **Module 3: Computer Vision with CNNs**
* **The Goal:** Re-attempt Fashion MNIST using **Convolutions**.
* **The Breakthrough:** I implemented **Conv2D** and **MaxPooling2D** layers.
    * *Convolutions* acted as filters to detect edges and shapes.
    * *Pooling* summarized the features, making the model translation-invariant.
* **The Outcome:** Much higher accuracy and the ability to visualize the internal "feature maps" of the network.

### **Module 4: The Final Boss - Real World Images**
* **The Goal:** Classify color images of **Horses vs. Humans**.
* **The Complexity:**
    * Images were different sizes, colors, and aspect ratios.
    * We used `ImageDataGenerator` to create a streaming pipeline.
    * We faced the **Overfitting** problem where the model memorized the training data but failed on validation.
* **The Solution:** I built a "Lightweight" model, reducing parameters and optimizing the architecture to achieve stable learning (~85% accuracy) without advanced regularization techniques.

---

## 📊 Key Results & Visualization

One of the most critical skills I learned was **interpreting training graphs** to diagnose model health.

* **Understanding Overfitting:** I learned that if Training Loss goes *down* but Validation Loss goes *up*, the model is memorizing, not learning.
* **The Fix:** By simplifying the model (reducing neurons from 512 to 64), I forced the network to learn general patterns instead of specific pixels.

---

## 🛠️ Tech Stack
* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy** & **Matplotlib** (for visualization)
* **Google Colab** (Development Environment)

## 🔮 What's Next?
This project establishes the foundation. My next steps in the **DeepLearning.AI TensorFlow Developer** specialization include:
1.  **Data Augmentation:** To solve the data scarcity problem in Module 4.
2.  **Transfer Learning:** Using pre-trained models (like InceptionV3) to achieve professional-grade accuracy.
3.  **NLP (Natural Language Processing):** Applying these neural network concepts to text.
4.  **Time Series:** Applying neural networks concepts to predict time series behavior.

---

### ✍️ Author
**[Victor Andrade Perone]**
*Machine Learning Engineer*


