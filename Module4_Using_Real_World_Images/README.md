# Module 4 - Using Real World Images

## Overview


> Hands-on exploration of building complex computer vision models to classify real-world images that are larger, more diverse, and less uniform than the MNIST datasets. This module focuses on the ImageDataGenerator for automated preprocessing and directory-based labeling.


This module transitions from simple, pre-processed datasets to handling raw image files stored in directories. You will learn to build models that can distinguish between complex subjects like "Horses vs. Humans" and "Happy vs. Sad" faces.

This module builds on CNN foundations and introduces:

- Handling image data from external directories
- Using the ImageDataGenerator for automatic labeling and rescaling
- Binary classification architectures
- Implementing validation sets to monitor model performance
- Advanced visualization of "information distillation" in deep layers
- Custom callbacks for early stopping based on accuracy thresholds



---

## Learning Objectives

In this module, you will learn:

- How to organize image data into subdirectories for automatic labeling
- How to use ImageDataGenerator to stream and rescale images from disk
- The role of the Sigmoid activation function in binary classification
- Why binary_crossentropy is used as a loss function for two-class problems
- How to design deeper CNN architectures to handle larger input shapes (e.g., 300×300 RGB)
- How to use validation data to check for overfitting during training
- How to visualize how a network "distills" raw pixels into abstract features

---

## Who This Module Is For

This module is designed for:

- Learners ready to move beyond "toy" datasets like MNIST
- Developers interested in building practical image classifiers from their own photos
- Students wanting to understand data pipelines and automated labeling in TensorFlow

---

## Skills Demonstrated

- Automated image preprocessing and normalization
- Building multi-layer Convolutional Neural Networks for high-resolution images
- Binary classification output modeling
- Data pipeline management using `flow_from_directory`
- Monitoring training progress with validation splits and custom callbacks

---

## Common Mistakes Explored in This Module

This module highlights common challenges encountered when working with real-world, non-uniform image data:

- **Incorrect Input Shapes:** Forgetting that real-world images must be resized to a uniform `target_size` before entering the network.
- **Mismatched Loss/Activation:** Using `softmax` or `categorical_crossentropy` for a single-unit binary output instead of `sigmoid` and binary_crossentropy.
- **Data Leakage:** Not properly separating training and validation directories.
- **Overfitting:** Training for too many epochs on a small dataset (like the Happy/Sad set) without monitoring validation accuracy.
- **Directory Structure Errors:** Misnaming subdirectories, which leads the `ImageDataGenerator` to mislabel the classes.

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
cd Module4_Using_Real_World_Images
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

1. Go to: [https://colab.research.google.com]
2. Click File → Open notebook
3. Select the GitHub tab
4. Paste your repository URL
5. Open Course_1_Part_8_Lesson_2_Notebook_Horses_Humans_Convnet.ipynb

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

- TensorFlow may produce informational or warning messages during execution.
- These messages do not affect the correctness of the exercises.
- CPU execution is sufficient for all notebooks in this module, though GPU is recommended for faster training on the 300x300 images.

---

## Reproducibility Note

Model training involves random initialization of weights.
As a result:
- Exact accuracy and loss values may vary slightly between runs
- Overall trends and conclusions should remain consistent

---

## Problem Statement

We are given datasets consisting of **real-world color images** (CGI) rather than the simple grayscale grids used in previous modules.

### Feature Extraction and Convolution Tasks

The core challenge in this module is `Feature Extraction`. Unlike simple digits, real-world images have complex backgrounds and subjects in varying positions.

- Convolution Layers: Must learn to identify specific features (e.g., the shape of a horse's ear vs. a human's ear) regardless of where they appear in the image.
- Pooling Layers: Must reduce the image size while preserving these essential features, effectively "summarizing" the presence of a feature in a region.
- Hierarchy: The network builds a hierarchy of features, starting from simple edges in early layers to complex body parts in deeper layers.


### Classification Tasks

The primary datasets used are **Horses vs. Humans** and **Happy vs. Sad**.

Unlike MNIST (28×28 grayscale), these images:

- Are significantly larger (e.g., 300×300 or 150×150 pixels)
- Are color images (3 color channels: Red, Green, Blue)
- Contain complex backgrounds and subjects in varying poses
- Are stored in raw image files (JPEG/PNG) rather than pre-packaged NumPy arrays

The goal is to train neural network models that can distinguish between two specific classes (Binary Classification).

### Automated Data Processing

In addition to classification, we explore how to handle data that doesn't fit into memory all at once.

- **Streaming:** Loading images from the disk in batches
- **Preprocessing:** Resizing non-uniform images to a target size on the fly
- **Labeling:** Inferring labels automatically based on directory names

### Overall Objective

By combining:

- `ImageDataGenerator` for efficient data pipelines
- Deeper Convolutional Neural Networks
- Binary classification outputs (Sigmoid activation)
- Validation sets to monitor overfitting

The aim is to build a robust pipeline for handling real-world image datasets.

---

## Dataset

### Horses vs. Humans

This dataset contains computer-generated images of horses and humans in various poses and backgrounds.

- **Training set:** ~1,000 images
- **Validation set:** ~250 images
- **Challenge:** The model must learn shapes and features specific to humans or horses, ignoring the complex backgrounds.

### Happy vs. Sad

A small dataset used for the final exercise containing 80 images of people's faces.

- **40 Happy faces**
- **40 Sad faces**
- **Challenge:** Training a model on very little data without overfitting.

---

## Data Preprocessing

Because real-world datasets are often too large to load into RAM, we use the `ImageDataGenerator` class provided by TensorFlow Keras.

### **Rescaling**

We normalize pixel values (0-255) to the range [0, 1] usually directly within the generator:

```python
train_datagen = ImageDataGenerator(rescale=1./255)
```

### **Flow from Directory**

Instead of manually writing loops to load files, we point the generator to a directory. It automatically:

- Finds images
- Resizes them to the target_size (e.g., 300×300)
- Assigns labels based on the folder name (e.g., images in .../horses/ get label 0, .../humans/ get label 1)

```python
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size=(300, 300),  # Resizes images to 300x300
    batch_size=128,          # Loads 128 images at a time for each training step
    class_mode='binary'      # Sets up for binary classification
)
```

---
## Model Architecture

To handle larger, more complex images, the CNN architecture becomes deeper than the one used for MNIST.

```python
model = tf.keras.models.Sequential([
    # First convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    # Output Layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Key Differences from Module 3

1. **Input Shape:** (300, 300, 3) includes 3 channels for color (RGB), significantly increasing the amount of data processed. 
2. **Depth:** We often stack 3, 4, or 5 convolutional layers to reduce the large image down to a manageable feature map size before the Flatten layer. 
3. **Output Layer:**
   1. **Module 3 (Multi-class):** Dense(10, activation='softmax')
   2. **Module 4 (Binary):** Dense(1, activation='sigmoid')

The `Sigmoid` activation forces the output to be a value between 0 and 1.

- Closer to 0: Class A (e.g., Horse)
- Closer to 1: Class B (e.g., Human)


This model serves as a **baseline**, treating images as flat vectors.

### Convolutional Neural Networks (CNNs)

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
- **Dense (1 neuron, Sigmoid output)**
  - Outputs a single probability score between 0 and 1.
  - Used specifically for binary classification.
  - **Thresholding:** Typically, values < 0.5 are predicted as Class 0, and values >= 0.5 are Class 1.

The number of convolutional filters and dense neurons is varied across experiments to demonstrate how **model capacity** affects feature learning and classification performance.

---

## Model Compilation

For binary classification, the optimizer and loss function change to reflect the nature of the output.

### Optimizer: RMSprop
While Adam is common, RMSprop is often preferred for automating the learning rate adjustment in recurrent or deep convolutional networks.

### Loss Function: Binary Crossentropy
Since we have only two classes and a single output neuron (0 to 1), we use binary_crossentropy instead of sparse_categorical_crossentropy.

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
```

---

## Training the Model

### Training with Generators

Since data is being streamed, we use the fit method with the generator object.

```python
history = model.fit(
    train_generator,
    steps_per_epoch=8,  # Total images = steps_per_epoch * batch_size
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
)
```

- **steps_per_epoch:** tells TensorFlow how many batches to draw from the generator to count as one "epoch".
- **validation_data:** allows us to pass a separate generator for validation images.

---

## Model Evaluation

This measures how well the model performs on **unseen data**.

Evaluating model performance in this module goes beyond just looking at the final accuracy score. We focus on two key aspects:

- **Training vs. Validation Accuracy:** We plot the accuracy history to check for overfitting.
  - **Good Model:** Training and Validation accuracy increase together.
  - **Overfitting:** Training accuracy hits 99-100%, but Validation accuracy stalls or decreases (e.g., remains at 80%).
- **Visualizing Intermediate Representations:** We use specific code blocks to visualize the "internal state" of the Convolutional layers. This allows us to see exactly what features (lines, shapes, textures) the model is activating on.

```python
# Example of accessing history for evaluation
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
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

### Binary Classification (Sigmoid) vs Multi-Class (Softmax)

| Feature | Binary Classification (Module 4) | Multi-Class Classification (Module 3) |
| :--- | :--- | :--- |
| **Example** | Horses vs Humans (2 classes) | Fashion MNIST (10 classes) |
| **Output Neurons** | 1 (Single value 0-1) | 10 (One per class) |
| **Activation** | `sigmoid` | `softmax` |
| **Loss Function** | `binary_crossentropy` | `sparse_categorical_crossentropy` |
| **Optimizer** | `RMSprop` (often preferred) | `Adam` (standard default) |
| **Input Shape** | `(300, 300, 3)` (High-res Color) | `(28, 28, 1)` (Low-res Grayscale) |
| **Output Meaning** | Probability of being Class 1 | Probability distribution across all classes |

In **Module 1**, the model predicts a single continuous number (regression).

In **Module 2**, the model predicts the probability that an input image belongs to each possible class (classification).

In **Module 3**, the model also performs classification, but uses convolutional neural networks to preserve spatial structure and learn visual features such as edges, textures, and shapes before producing a probability distribution over classes.

In **Module 4**, **Real-World Binary Classification**, Single probability score (0-1) from high-res color data

### Model Capacity

Overfitting occurs when the model learns the training data too well but fails to generalize. Common causes explored in this module:

- Too many neurons or layers (Excessive Capacity)
- Training for too many epochs

Later exercises introduce techniques to control training behavior instead of blindly increasing epochs.

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
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True
```

---

## Exercise Progression (Conceptual Overview)

The exercises in this module are designed to progressively introduce computer vision concepts by moving from simple models to more complex architectures and from automated to manual implementations.

| File Name | Task & Goal | Key Learning |
| :--- | :--- | :--- |
| **Lesson 2: Horses vs Humans** | Train a CNN from scratch on complex CGI images. | Using `ImageDataGenerator` and building deep CNNs. |
| **Lesson 3: With Validation** | Add a validation set to the training loop. | Identifying **overfitting** when training on small datasets. |
| **Lesson 4: Compact Images** | Train on smaller/compressed images. | Understanding the trade-off between image resolution and training speed. |
| **Semana_4_Exercicio** | **Assignment:** Happy vs Sad Face detection. | Using **Callbacks** to stop training automatically when accuracy > 99.9%. |
---

## Summary

This module demonstrates how neural networks can:

- Learn visual patterns from image data
- Classify inputs into categories
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
📁 Module4-Using-Real-World-Images
├── 📓 Course_1_Part_8_Lesson_2_Notebook_Horses_Humans_Convnet.ipynb
├── 📓 Course_1_Part_8_Lesson_3_Notebook_Horses_Humans_with_Validation.ipynb
├── 📓 Course_1_Part_8_Lesson_4_Notebook_Horses_Humans_Compact_Images.ipynb
├── 📓 Semana_4_Exercicio.ipynb
├── 📄 Exercise4-Question.json
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

- [TensorFlow ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator) – Official documentation for data augmentation and preprocessing.
- [Binary Crossentropy](https://keras.io/api/losses/probabilistic_losses/) – Details on the loss function used for binary classification.
- [Understanding Overfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit?hl=pt-br) – Strategies to prevent models from memorizing data instead of learning features.
- [Visualizing CNN Features](https://distill.pub/2017/feature-visualization/) – An in-depth look at what neural networks actually "see".
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) – Official API reference and tutorials.
- [Keras Documentation](https://keras.io/) – Guides for layers, models, and training.
- [Understanding Loss and Accuracy](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/) – Explains why loss and accuracy are different.
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn) - Official TensorFlow guide for building and training convolutional neural networks using Keras.
- [Pooling in Convolutional Neural Networks](https://www.digitalocean.com/community/tutorials/pooling-in-convolutional-neural-networks) - A practical tutorial explaining pooling operations like max and average pooling and why they are useful. 
- [Convolutional Neural Network (CNN) – Wikipedia](https://en.wikipedia.org/wiki/Convolutional_neural_network) – Encyclopedic overview of CNN architecture including convolution and pooling concepts.
- [Pooling layer – Wikipedia](https://en.wikipedia.org/wiki/Pooling_layer) – Focused article on pooling layers, detailing how pooling reduces spatial dimensions in CNNs.
- [Python CNN with TensorFlow Tutorial](https://www.datacamp.com/tutorial/cnn-tensorflow-python) – Step-by-step beginner tutorial on CNNs and pooling using TensorFlow code examples. 

