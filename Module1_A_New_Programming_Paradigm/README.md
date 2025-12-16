# Module 1 - A New Programming Paradigm

## Overview

This module introduces a new way of programming using machine learning, specifically with TensorFlow and Keras.

Instead of explicitly defining rules and equations, we train a model to learn the relationship between inputs and outputs from data.

This is designed for learners with no prior experience in TensorFlow or machine learning.

---

## Learning Objective

This example shows:
- What it means to program using data instead of rules
- How a simple neural network works
- How to train a model to predict new values
- The basic TensorFlow/Keras workflow:
  - Define a model
  - Compile it
  - Train it
  - Make predictions

---

## Problem Statement

We are given a set of input values (xs) and corresponding outputs (ys):

| Input (xs)  | Output (ys)  |
| ---------  | ----------  |
| 1.0        | 1.0         |
| 2.0        | 1.5         |
| 3.0        | 2.0         |
| 4.0        | 2.5         |
| 5.0        | 3.0         |
| 6.0        | 3.5         |
| 7.0        | 4.0         |
| 8.0        | 4.5         |
| 9.0        | 5.0         |
| 10.0       | 5.5         |
| 11.0       | 6.0         |
| 12.0       | 6.5         |
| 13.0       | 7.0         |
| 14.0       | 7.5         |


The relationship between `xs` and `ys` is not explicitly programmed.
Instead, we want the model to learn the pattern and then predict the output for a new input value.

In traditional programming, we would try to manually define a mathematical rule such as:

y = 0.5x + 0.5

However, in this example, **we do not provide this formula to the program**.

Instead, we give the model:
- A set of input values (`xs`)
- Their corresponding correct outputs (`ys`)

The goal of the model is to **learn the underlying mathematical relationship** between `x` and `y` directly from the data.

Once trained, the model should be able to **generalize** this learned relationship and predict the output for new, unseen input values — for example:

x = 7.0 → y ≈ 4.0

---

## Key Concept: A New Programming Paradigm

Traditional programming:

Input + Rules → Output

Machine Learning:

Input + Output → Rules (Model)

In this paradigm:
- We provide examples
- The model learns the underlying relationship
- We use the trained model to make predictions on unseen data

---

## Dataset

- `xs` represents the input values
- `ys` represents the expected outputs
- The model will learn how `y` changes as `x` increases

```python
xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],dtype=float)
ys = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5],dtype=float)
```

---

## Model Architecture

We use a very simple neural network:
- A single dense (fully connected) layer
- One input neuron
- One output neuron

This is effectively learning a linear relationship between x and y.


Internally, this single neuron computes an equation of the form:

```
y = w · x + b
```

Where:
- `w` is the weight
- `b` is the bias

At the beginning of training, the values of `w` and `b` are random.
During training, the model automatically adjusts these values so that the predicted output gets closer to the true output.

In other words, instead of manually choosing the values for `w` and `b`, we let the model **learn them from data**.

Given enough training, the learned values of `w` and `b` will approximate the linear rule that best fits the dataset.


```python
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
```

---

## Model Compilation

- Optimizer (SGD): Adjusts the model’s weights to reduce error
- Loss function (MSE): Measures how far predictions are from actual values

```python
model.compile(
    optimizer='sgd',
    loss='mean_squared_error'
)
```
### Optimizer: Stochastic Gradient Descent (SGD)
The optimizer is responsible for updating the model’s internal parameters (weights and bias) during training.
- Stochastic Gradient Descent (SGD) adjusts the weights step by step
- After each iteration, it moves the weights in the direction that reduces the error
- Over time, this process helps the model converge toward a better approximation of the true relationship

### Loss Function: Mean Squared Error (MSE)

The loss function measures how well the model is performing.

Mean Squared Error computes the average of the squared differences between the predicted values and the true values:

```
MSE = (1 / n) Σ (y_true − y_pred)²
```

- Larger errors are penalized more heavily
- Squaring ensures all errors are positive
- The goal of training is to minimize this loss

During training, the optimizer uses the loss value to decide how to update the model’s parameters.

---

## Training the Model

### Conceptual Summary

At this point, the model represents a simple function:

- Input: a single numerical value (`x`)
- Output: a predicted numerical value (`y`)
- Learnable parameters: `w` (weight) and `b` (bias)

Training consists of finding the values of `w` and `b` that best map inputs to outputs.

- The model sees the data 500 times
- Each epoch improves the model’s understanding of the relationship
- More epochs → better approximation (up to a point)

An epoch corresponds to one complete pass through the entire training dataset.


```python
model.fit(xs, ys, epochs=500)
```

---

## Making a prediction

After training, we ask the model to predict a new value:

```python
# Wrap 7.0 in an array because the model expects a batch of inputs
print(model.predict(np.array([7.0])))

```

The expected output should be close to 4.0, even though the model has never seen 7.0 before.

This demonstrates generalization — one of the core goals of machine learning.

---

## Why Isn’t the Result Exactly 4.0?

- The model starts with random weights
- Training is an approximation process
- With a simple dataset and model, small deviations are expected

This is normal and expected behavior in machine learning.

---

## Files in This Module

```
📁 Module1_A_New_Programming_Paradigm
├── 📓 Exercise_1_House_Prices_Question.ipynb
└── 📄 requirements.txt
└── 📄 README.md
```
