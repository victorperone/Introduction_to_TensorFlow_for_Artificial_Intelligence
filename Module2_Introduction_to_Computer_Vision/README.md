# Module 2 - Introduction to Computer Vision

## Overview


> Hands-on exploration of image classification using TensorFlow and Keras, with focus on model architecture, overfitting, and training control.


This module introduces computer vision using neural networks with TensorFlow and Keras.

Instead of working with simple numeric values, we now train a model to recognize handwritten digits from images using the MNIST dataset.

This module builds on the concepts from Module 1 and introduces:

- Image data
- Classification problems
- Multi-layer neural networks
- Softmax outputs for probabilities



---

## Learning Objective

In this module, you will learn:

- How image data is represented as numbers
- How to normalize image pixel values
- How to build a neural network for multi-class classification
- How to train and evaluate a model on image data
- How model architecture affects performance
- How training for too long can lead to overfitting
- How to interpret prediction probabilities
- How callbacks can be used to control training

---

## Common Mistakes Explored in This Module

This module intentionally includes experiments that demonstrate common beginner mistakes:

- Using an incorrect output layer size
- Training for too many epochs
- Increasing model size without improving generalization
- Assuming higher accuracy always means a better model

Understanding these mistakes is essential for building reliable machine learning systems.

---

## How to Run

This module consists of Jupyter notebooks that can be run locally.

### Prerequisites

- Python 3.8 or higher
- pip
- Virtual environment support (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/victorperone/Introduction_to_TensorFlow_for_Artificial_Intelligence.git
cd Module2_Introduction_to_Computer_Vision
```

### 2. Create and Activate a Virtual Environment (Recommended)

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

All required packages are listed in `requirements.txt`

```python
pip install -r requirements.txt
```
This will install TensorFlow and other necessary libraries.

### 4. Launch Jupyter Notebook

You can run the notebooks **locally** or using **Google Colab**.

#### Option A: Run Locally (Jupyter Notebook)

```bash
jupyter notebook
```

or, if you prefer JupyterLab:
```bash
jupyter lab
```

### Option B: Run on Google Colab (No Local Setup Required)

1. Go to: https://colab.research.google.com
2. Click File → Open notebook
3. Select the GitHub tab
4. Paste your repository URL
5. Open Course_1_Part_4_Lesson_2_Notebook.ipynb

Google Colab provides:

- Free CPU (and optional GPU) execution
- No local Python or TensorFlow installation
- Automatic dependency handling for most libraries

⚠️ Note: If requirements.txt is not automatically handled, install dependencies in a Colab cell:

```python
!pip install -r requirements.txt
```

### 5. Run the Exercises

Open the notebooks in numerical order
Run each cell sequentially
Observe how changes in model architecture, training duration, and callbacks affect results
It is recommended to run the exercises **in order**, as each one builds conceptually on the previous examples.

### Environment Notes

- TensorFlow may produce informational or warning messages during execution
- These messages do not affect the correctness of the exercises
- CPU execution is sufficient for all notebooks in this module

---

## Reproducibility Note

Model training involves random initialization of weights.
As a result:
- Exact accuracy and loss values may vary slightly between runs
- Overall trends and conclusions should remain consistent

---

## Problem Statement

We are given a dataset of handwritten digit images (0–9).

Each image:

- Is 28 × 28 pixels
- Is grayscale (values from 0 to 255)
- Represents a single digit

The goal is to train a model that can correctly classify an image as one of the digits 0 through 9.

This is a classification problem, not a regression problem.

---

## Dataset

### MNIST
The MNIST dataset is included directly in TensorFlow:

```python
mnist = tf.keras.datasets.mnist
```

It contains:

- **60,000 training images**
- **10,000 test images**

Each image is paired with a label:
- 0 → digit zero
- 1 → digit one
- …
- 9 → digit nine

### Loading the data

```python
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```

### Fashion-MNIST

It contains:

- Clothing categories (e.g. T-shirt, trousers, shoes)
- Same image size and structure as MNIST
- More visually complex classification task

```python
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```

---

## Data Preprocessing

Neural networks work best when input values are small.

We normalize pixel values by dividing by 255:

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

This scales pixel values to the range [0, 1].

---
## Model Architecture

We use a **feedforward neural network** with the following layers:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### Layer Explanation

- **Flatten**
  - To convert images into vectors
  - Converts the 28×28 image into a 1D vector (784 values)
- **Dense (fully connected layers) with ReLU activation**
  - Learns complex patterns in the image
- **Dense (10 neurons, Softmax) output**
  - Outputs probabilities for each digit (0–9)

The number of layers and neurons is varied to show how model capacity affects learning.

---

## Model Compilation

### Optimizer: Adam
- Adaptive learning rate
- Faster convergence than basic SGD
- Common default choice for deep learning

### Loss Function: Sparse Categorical Crossentropy
- Used for multi-class classification
- Labels are integers (0–9), not one-hot encoded

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy'
    )
```

Some exercises include:
- More epochs
- Larger networks
- Accuracy metrics
- Training callbacks

---

## Training the Model

- The model sees the training data 5 times
- Each epoch improves classification accuracy
- More epochs generally improve performance (up to a point)

```python
model.fit(training_images, training_labels, epochs=5)
```

---

## Model Evaluation

This measures how well the model performs on **unseen data**.
```python
model.evaluate(test_images, test_labels)
```

---

## Model Evaluation Metrics

Throughout this module, different metrics are used to evaluate model performance.
The two most important ones are **loss** and **accuracy**.

Although they are related, they measure **different aspects** of model behavior.

### Loss

**Loss** measures how far the model’s predictions are from the correct answers.

For classification tasks in this module, we use **Sparse Categorical Crossentropy** as the loss function.

Conceptually, this loss:
- Compares the predicted probability distribution with the true label
- Penalizes confident but wrong predictions more heavily
- Produces a continuous numerical value (lower is better)

Simplified intuition:
- Predicting the correct class with **high confidence** → low loss
- Predicting the correct class with **low confidence** → higher loss
- Predicting the wrong class with **high confidence** → very high loss

Loss is the value the **optimizer actively tries to minimize** during training.

### Accuracy

**Accuracy** measures how often the model’s final prediction is correct.

It is calculated as:

```
accuracy = (number of correct predictions) / (total number of predictions)
```

Accuracy only checks:
- Whether the predicted class (highest probability) matches the true label

It does **not** consider:
- How confident the model was
- How wrong an incorrect prediction was

### Why Loss and Accuracy Are Different

Loss and accuracy can improve at different rates.

Examples:
- Accuracy may stay the same while loss decreases
  → The model is becoming **more confident** in its correct predictions
- Accuracy may increase while loss remains relatively high
  → Some predictions are still made with low confidence
- Loss may increase even when accuracy is high
  → A few very confident wrong predictions are heavily penalized

Because of this:
- **Loss** is better for guiding training
- **Accuracy** is better for understanding overall correctness

---

### Metrics in This Module

- Most exercises monitor **loss**
- Later exercises explicitly track **accuracy**
- Some callbacks stop training when:
  - Loss drops below a threshold
  - Accuracy exceeds a target value

Using both metrics together provides a more complete picture of model performance.

---

### Key Takeaway

- **Loss tells the optimizer how to learn**
- **Accuracy tells humans how well the model performs**

Both are necessary to properly evaluate a classification model.


---

## Making a prediction

After training, we ask the model to predict a new value:

```python
classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])

```

- The model outputs **probabilities** for each digit
- The highest probability corresponds to the predicted digit
- Comparing prediction vs label helps validate correctness

---

## Key Concepts

### Classification vs Regression

| Module   | Task Type        | Output                         |
|----------|------------------|--------------------------------|
| Module 1 | Regression       | Single numeric value           |
| Module 2 | Classification   | Probability distribution       |

In **Module 1**, the model predicts a single continuous number (regression).

In **Module 2**, the model predicts the probability that an input image belongs to each possible class (classification).

### Model Capacity

As the number of neurons and layers increases:
- The model can learn more complex patterns
- Training accuracy often improves
- The risk of overfitting increases

Small models may **underfit**
Large models may **overfit**

### Overfitting

Overfitting occurs when:

- The model learns the training data too well
- It fails to generalize to unseen data

Common causes explored in this module:

- Too many neurons
- Too many layers
- Too many training epochs

Later exercises introduce techniques to control training behavior instead of blindly increasing epochs.

---

## Early Stopping with Callbacks

Callbacks monitor training metrics and:

- Stop training when performance is “good enough”
- Prevent unnecessary epochs
- Reduce overfitting risk

This is an important step toward **production-ready models**.

```python
tf.keras.callbacks.Callback
```

---

## Exercise Progression (Conceptual Overview)

The exercises in this module are designed to progressively introduce new concepts by
modifying model architecture, training duration, datasets, and training control mechanisms.

| Exercise | Dataset          | Architecture Focus                     | Training Focus                | Key Concept Demonstrated                  |
|----------|------------------|----------------------------------------|-------------------------------|-------------------------------------------|
| 1        | MNIST            | Basic network (Flatten + Dense)        | Short training (5 epochs)     | Baseline image classification             |
| 2        | MNIST            | Missing / altered input structure      | Same training setup           | Importance of correct input shape         |
| 3        | MNIST            | Incorrect output layer size            | Same training setup           | Output layer must match number of classes |
| 4        | MNIST            | Deeper network (multiple Dense layers) | Same training setup           | Increased model capacity                  |
| 5        | MNIST            | Moderate-sized network                 | Long training (30 epochs)     | Overfitting risk from excessive epochs    |
| 6        | MNIST            | Large hidden layer (512 neurons)       | Short training                | Capacity vs generalization tradeoff       |
| 7        | Fashion-MNIST    | Deep network                           | Callback-based early stopping | Controlling training dynamically          |
| 8        | Fashion-MNIST    | Deeper & wider network                 | Accuracy-based early stopping | Preventing overfitting with callbacks     |

This progression demonstrates that improving a model is not only about adding more layers or epochs,
but about balancing **model complexity**, **training time**, and **generalization performance**.

---

## Summary

This module demonstrates how neural networks can:

- Learn visual patterns from image data
- Classify inputs into multiple categories
- Output confidence scores for predictions

It represents a major step forward from simple numeric prediction to real-world machine learning applications like image recognition.

---

## Files in This Module

```
📁 Module2_Introduction_to_Computer_Vision
├── 📓 Course_1_Part_4_Lesson_2_Notebook.ipynb
└── 📄 requirements.txt
└── 📄 README.md
```

---

## Limitations

- Models use fully connected layers instead of convolutional layers
- No explicit validation split is used in most exercises
- Performance is not optimized for real-world deployment

These limitations are intentional and will be addressed in later modules.

---

## Further Reading

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) – Official API reference and tutorials.
- [Keras Documentation](https://keras.io/) – Guides for layers, models, and training.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) – Original dataset page with examples and download.
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) – Dataset page with labels and examples.
- [Understanding Loss and Accuracy](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/) – Explains why loss and accuracy are different.

