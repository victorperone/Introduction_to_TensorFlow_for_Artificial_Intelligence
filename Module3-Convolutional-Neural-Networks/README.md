# Module 3- Convolutional Neural Networks (CNNs)

## Overview


> Hands-on exploration of image classification and convolution using TensorFlow, Keras, and NumPy, with a focus on feature extraction, model architecture, and visual understanding of CNNs.


This module introduces computer vision with convolutional neural networks (CNNs).

Instead of working with simple numeric values, we train models to recognize images using the MNIST and Fashion-MNIST datasets, and we also manually implement convolution and pooling to understand how CNNs work internally.

This module builds on earlier neural network concepts and introduces:

- Image data and spatial structure
- Convolution and pooling layers
- Feature extraction
- CNN-based classification
- Visualization of learned filters
- Manual implementation of convolution and pooling



---

## Learning Objectives

In this module, you will learn:

- How image data is represented as multidimensional arrays
- How to normalize image pixel values
- The difference between dense networks and convolutional neural networks
- How convolutional layers extract spatial features
- How pooling layers reduce dimensionality
- How to train and evaluate CNN models
- How to visualize intermediate CNN feature maps
- How convolution and pooling work at a low level using NumPy

---

## Who This Module Is For

This module is designed for:
- Beginners transitioning from basic neural networks to computer vision
- Learners new to convolution and pooling concepts
- Students building intuition before advanced CNN architectures

---

## Skills Demonstrated

- Image preprocessing and normalization
- Dense neural networks for image classification
- Convolutional neural networks with TensorFlow and Keras
- Feature extraction and spatial reasoning
- Manual implementation of convolution and pooling
- Model evaluation using loss and accuracy

---

## Common Mistakes Explored in This Module

This module intentionally includes experiments and comparisons that highlight common beginner mistakes when working with image data and convolutional neural networks:

- Treating image data as flat vectors too early and losing spatial information
- Forgetting to reshape image data correctly for convolutional layers
- Using convolutional layers without understanding what features they extract
- Increasing the number of filters or layers without clear performance gains
- Assuming deeper CNNs always outperform simpler architectures
- Misinterpreting accuracy without examining model behavior visually

In addition, the manual convolution exercises demonstrate how easy it is to:

- Misapply convolution filters
- Ignore boundary conditions
- Misunderstand the effect of pooling on spatial resolution

Understanding these mistakes helps build intuition about **why CNNs work**, not just how to use them.

---

## How to Run

This module consists of Jupyter notebooks that can be run locally or on Google Colab.

### Prerequisites

- Python 3.8 or higher
- pip
- Virtual environment support (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/victorperone/Introduction_to_TensorFlow_for_Artificial_Intelligence.git
cd Module3_Convolutional_Neural_Networks
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

We are given grayscale image datasets used for visual recognition tasks.

### Classification Tasks

The primary datasets used are **MNIST** and **Fashion-MNIST**.

Each image:

- Is 28 × 28 pixels
- Is grayscale (values from 0 to 255)
- Represents a single class:
    - Digits (0–9) for MNIST
    - Clothing categories for Fashion-MNIST

The goal is to train neural network models that can correctly classify each image into one of 10 possible classes.

This is a multi-class classification problem, not a regression problem.

### Feature Extraction and Convolution Tasks

In addition to classification, we also explore how convolution and pooling work internally.

Given a grayscale image:

- Apply a 3×3 convolution filter manually
- Highlight edges and structural features
- Reduce spatial dimensions using max pooling

The goal of this part is **conceptual understanding**, not model accuracy.

### Overall Objective

By combining:

- Dense neural networks
- Convolutional neural networks
- Visualization of learned features
- Manual convolution and pooling

The aim is to understand both the theory and mechanics behind modern computer vision models.

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

We normalize pixel values by dividing by 255 to improve stability:

```python
training_images = training_images / 255.0
test_images = test_images / 255.0
```

This scales pixel values to the range [0, 1].


For CNN models, images are reshaped to include a channel dimension:

```python
training_images = training_images.reshape(60000, 28, 28, 1)
```
This represents grayscale images with a single color channel.

---
## Model Architecture

The module progresses from dense models to convolutional models to clearly demonstrate why spatial structure matters in image classification.

CNNs are introduced after a dense baseline to highlight the importance of spatial structure in image data.

### Dense Baseline Model

We use a **feedforward neural network** with the following layers:

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

This model serves as a **baseline**, treating images as flat vectors.

### Convolutional Neural Networks (CNNs)

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

CNNs preserve spatial structure and learn visual features such as edges and shapes.

- **Intuition:** Dense networks see pixels; CNNs see patterns.

### Layer Explanation

- **Conv2D (Convolutional Layers with ReLU Activation)**: Convolutional layers are the core building blocks of CNNs. Instead of connecting every pixel to every neuron, they operate on small local regions of the image.
  - Applies multiple learnable 3×3 filters (kernels) that slide across the image
  - Each filter performs a dot product between its weights and a small patch of the image
  - The result of each filter is a feature map highlighting where a specific pattern appears
  - Early convolutional layers typically learn simple features:
    - Edges (horizontal, vertical, diagonal)
    - Corners
    - Basic textures
  - Deeper convolutional layers combine earlier features to detect more complex patterns:
    - Shapes
    - Object parts
    - High-level visual structures
  - Weight sharing means the same filter is applied across the entire image:
    - Greatly reduces the number of parameters
    - Makes the model efficient and scalable
  - ReLU activation:
    - Keeps positive feature responses
    - Removes negative values
    - Introduces non-linearity so the model can learn complex patterns
  - **🔑 Key intuition:**
    - Convolutional layers learn what to look for and where it appears in the image.

- **MaxPooling2D**: Pooling layers reduce the size of feature maps while retaining the most important information.
  - Operates on small windows (commonly 2×2) of each feature map
  - Replaces each window with its maximum value
  - Reduces spatial dimensions (width and height) by a factor of 2
  - Keeps the strongest feature activations while discarding weaker ones
  - Makes the network:
    - Faster (fewer computations)
    - Less memory-intensive
    - More resistant to small shifts or distortions in the image
  - Helps prevent the network from focusing too much on exact pixel locations
  - **🔑 Key intuition:**
    - Pooling answers whether a feature is present, not exactly where it is.

- **Flatten**
  - Converts the 2D feature maps into a 1D vector
  - Prepares convolutional features for the dense layers
  - **Dense (fully connected layer with ReLU activation)**
    - Learns higher-level combinations of extracted features
    - Acts as a classifier based on convolutional features
  - **Dense (10 neurons, Softmax output)**
    - Outputs a probability distribution over the 10 classes
    - The class with the highest probability is the model’s prediction

The number of convolutional filters and dense neurons is varied across experiments to demonstrate how **model capacity** affects feature learning and classification performance.

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
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
```

---

## Training the Model

- The model sees the training data 5 times
- Each epoch improves classification accuracy
- More epochs generally improve performance (up to a point)

```python
model.fit(training_images, training_labels, epochs=5)
```

Models are trained for 5-10 epochs, balancing learning and overfitting risk.

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

| Module   | Task Type                   | Output                                                 |
|----------|-----------------------------|--------------------------------------------------------|
| Module 1 | Regression                  | Single numeric value                                   |
| Module 2 | Classification              | Probability distribution                               |
| Module 3 | Image Classification (CNNs) | Probability distribution with spatial feature learning |

In **Module 1**, the model predicts a single continuous number (regression).

In **Module 2**, the model predicts the probability that an input image belongs to each possible class (classification).

In **Module 3**, the model also performs classification, but uses convolutional neural networks to preserve spatial structure and learn visual features such as edges, textures, and shapes before producing a probability distribution over classes.

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

In this module, callbacks are introduced conceptually; more advanced usage is explored in later modules.

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

The exercises in this module are designed to progressively introduce computer vision concepts by moving from simple models to more complex architectures and from automated to manual implementations.

| Exercise | Dataset       | Architecture Focus                  | Training Focus          | Key Concept Demonstrated                   |
| -------- | ------------- | ----------------------------------- | ----------------------- | ------------------------------------------ |
| 1        | Fashion-MNIST | Dense network (Flatten + Dense)     | Short training (5 ep)   | Baseline image classification              |
| 2        | Fashion-MNIST | CNN with Conv2D + MaxPooling        | Short training (5 ep)   | Spatial feature extraction                 |
| 3        | Fashion-MNIST | Deeper CNN (multiple Conv2D layers) | Same training setup     | Improved feature learning                  |
| 4        | Fashion-MNIST | CNN with different filter sizes     | Same training setup     | Model capacity vs performance              |
| 5        | Fashion-MNIST | CNN feature-map visualization       | No training change      | Understanding learned filters              |
| 6        | MNIST         | CNN applied to digit recognition    | Longer training (10 ep) | Dataset complexity comparison              |
| 7        | SciPy Image   | Manual convolution (3×3 filters)    | No learning step        | How convolution works internally           |
| 8        | SciPy Image   | Manual max pooling                  | No learning step        | Spatial downsampling and feature dominance |


---

## Summary

This module demonstrates how neural networks can:

- Learn visual patterns from image data
- Classify inputs into multiple categories
- Output confidence scores for predictions

It represents a major step forward from simple numeric prediction to real-world machine learning applications like image recognition.

### Key Takeaways

- Convolutional Neural Networks (CNNs) are better suited for image data than dense-only networks because they preserve spatial relationships.
- Convolutional layers learn local visual features such as edges, textures, and shapes.
- Pooling layers reduce spatial dimensions while retaining important information.
- Dense layers act as classifiers on top of extracted convolutional features.
- Visualizing intermediate feature maps helps explain what CNNs learn internally.
- Manual convolution and pooling provide intuition about how CNN layers operate mathematically.
- Increasing model complexity does not always lead to better generalization.
- Understanding model behavior is as important as achieving high accuracy.

---

## Files in This Module

```
📁 Module3-Convolutional_Neural_Networks
├── 📓 Course_1_Part_6_Lesson_2_Notebook.ipynb
└── 📓 Course_1_Part_6_Lesson_3_Notebook.ipynb
└── 📄 requirements.txt
└── 📄 README.md
```

---

## Limitations

- Models are trained for demonstration, not optimization
- No explicit validation split is used
- Manual convolution is educational, not performance-oriented

These limitations are intentional and will be addressed in later modules.

---

## Further Reading

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) – Official API reference and tutorials.
- [Keras Documentation](https://keras.io/) – Guides for layers, models, and training.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) – Original dataset page with examples and download.
- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) – Dataset page with labels and examples.
- [Understanding Loss and Accuracy](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/) – Explains why loss and accuracy are different.
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn) - Official TensorFlow guide for building and training convolutional neural networks using Keras.
- [Pooling in Convolutional Neural Networks](https://www.digitalocean.com/community/tutorials/pooling-in-convolutional-neural-networks) - A practical tutorial explaining pooling operations like max and average pooling and why they are useful. 
- [Convolutional Neural Network (CNN) – Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) – Encyclopedic overview of CNN architecture including convolution and pooling concepts.
- [Pooling layer – Wikipedia](https://en.wikipedia.org/wiki/Pooling_layer) – Focused article on pooling layers, detailing how pooling reduces spatial dimensions in CNNs.
- [Python CNN with TensorFlow Tutorial](https://www.datacamp.com/tutorial/cnn-tensorflow-python) – Step-by-step beginner tutorial on CNNs and pooling using TensorFlow code examples. 


