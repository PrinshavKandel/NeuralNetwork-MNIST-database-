# MNIST Digit Classifier — Neural Network from Scratch

A fully functional **2-layer neural network** built using only `NumPy` and `Pandas` that classifies handwritten digits (0–9) from the [MNIST dataset](https://www.kaggle.com/c/digit-recognizer). No deep learning frameworks — just math.





# Network Architecture

```
Input Layer        Hidden Layer       Output Layer
  (784)        →      (10)         →      (10)
 pixels            ReLU activation    Softmax activation
```

| Layer | Size | Activation |
|-------|------|------------|
| Input | 784 (28×28 pixels) | — |
| Hidden | 10 neurons | ReLU |
| Output | 10 neurons (digits 0–9) | Softmax |

---

##  Setup & Usage

### Requirements
```bash
pip install numpy pandas matplotlib
```



Make sure `train.csv` is available at the correct path (default: `/kaggle/input/digit-recognizer/train.csv`). Update this path if running locally.

---

# Data Preprocessing

The dataset is loaded, shuffled, and split into:
- **Validation set**: first 1,000 samples
- **Training set**: remaining samples

Each image is a flattened vector of 784 pixel values (28×28), normalized to the range [0, 1]:

$$X = \frac{X_{\text{raw}}}{255}$$

The data matrix $X$ has shape **(784, m)** where **m** is the number of samples, so each **column** is one image.

---

# Weight Initialization

Weights and biases are initialized randomly in the range $[-0.5, 0.5]$:

$$W_{\text{hidden}} \in \mathbb{R}^{10 \times 784}, \quad b_{\text{hidden}} \in \mathbb{R}^{10 \times 1}$$

$$W_{\text{output}} \in \mathbb{R}^{10 \times 10}, \quad b_{\text{output}} \in \mathbb{R}^{10 \times 1}$$

```python
W_hidden = np.random.rand(10, 784) - 0.5
b_hidden = np.random.rand(10, 1)  - 0.5
W_output = np.random.rand(10, 10) - 0.5
b_output = np.random.rand(10, 1)  - 0.5
```

---

#Forward Propagation

Forward propagation computes the network's prediction by passing the input through each layer.

### Step 1 — Hidden Layer Pre-activation

$$Z_{\text{hidden}} = W_{\text{hidden}} \cdot X + b_{\text{hidden}}$$

- $W_{\text{hidden}}$: shape $(10, 784)$
- $X$: shape $(784, m)$
- $b_{\text{hidden}}$: shape $(10, 1)$ — broadcast across all $m$ samples
- $Z_{\text{hidden}}$: shape $(10, m)$

### Step 2 — Hidden Layer Activation (ReLU)

$$A_{\text{hidden}} = \text{ReLU}(Z_{\text{hidden}}) = \max(0,\ Z_{\text{hidden}})$$

ReLU zeroes out all negative values, introducing non-linearity.

### Step 3 — Output Layer Pre-activation

$$Z_{\text{out}} = W_{\text{output}} \cdot A_{\text{hidden}} + b_{\text{output}}$$

- $W_{\text{output}}$: shape $(10, 10)$
- $A_{\text{hidden}}$: shape $(10, m)$
- $Z_{\text{out}}$: shape $(10, m)$

### Step 4 — Output Layer Activation (Softmax)

$$A_{\text{out}} = \text{softmax}(Z_{\text{out}}) = \frac{e^{Z_{\text{out}}}}{\sum_{k=0}^{9} e^{Z_{\text{out},k}}}$$

Each column of $A_{\text{out}}$ is a probability distribution over the 10 digit classes, summing to 1.

---

# Backward Propagation

Backpropagation computes gradients of the loss with respect to every parameter using the **chain rule**, so we can update weights to reduce error.

### Loss Function — Categorical Cross-Entropy

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=0}^{9} Y_k^{(i)} \log\left(A_{\text{out},k}^{(i)}\right)$$

where $Y$ is the **one-hot encoded** label matrix of shape $(10, m)$.

### One-Hot Encoding

$$Y_{\text{one-hot}} \in \{0, 1\}^{10 \times m}, \quad Y_{\text{one-hot},k}^{(i)} = \begin{cases} 1 & \text{if label}^{(i)} = k \\ 0 & \text{otherwise} \end{cases}$$

---

### Step 1 — Output Layer Gradients

The gradient of the loss w.r.t. $Z_{\text{out}}$ (combined softmax + cross-entropy derivative):

$$\frac{\partial \mathcal{L}}{\partial Z_{\text{out}}} = dZ_{\text{out}} = A_{\text{out}} - Y_{\text{one-hot}}$$

Weight and bias gradients for the output layer:

$$\frac{\partial \mathcal{L}}{\partial W_{\text{output}}} = dW_{\text{output}} = \frac{1}{m}\ dZ_{\text{out}} \cdot A_{\text{hidden}}^T$$

$$\frac{\partial \mathcal{L}}{\partial b_{\text{output}}} = db_{\text{output}} = \frac{1}{m} \sum_{i=1}^{m} dZ_{\text{out}}^{(i)}$$

Matrix shapes:
- $dZ_{\text{out}}$: $(10, m)$
- $A_{\text{hidden}}^T$: $(m, 10)$
- $dW_{\text{output}}$: $(10, 10)$ ✓

---

### Step 2 — Hidden Layer Gradients

Backpropagate the error through the output weights:

$$\frac{\partial \mathcal{L}}{\partial Z_{\text{hidden}}} = dZ_{\text{hidden}} = W_{\text{output}}^T \cdot dZ_{\text{out}}\ \odot\ \text{ReLU}'(Z_{\text{hidden}})$$

where $\odot$ is element-wise multiplication and the **ReLU derivative** is:

$$\text{ReLU}'(Z) = \begin{cases} 1 & \text{if } Z > 0 \\ 0 & \text{if } Z \leq 0 \end{cases}$$

Weight and bias gradients for the hidden layer:

$$\frac{\partial \mathcal{L}}{\partial W_{\text{hidden}}} = dW_{\text{hidden}} = \frac{1}{m}\ dZ_{\text{hidden}} \cdot X^T$$

$$\frac{\partial \mathcal{L}}{\partial b_{\text{hidden}}} = db_{\text{hidden}} = \frac{1}{m} \sum_{i=1}^{m} dZ_{\text{hidden}}^{(i)}$$

Matrix shapes:
- $W_{\text{output}}^T$: $(10, 10)$
- $dZ_{\text{out}}$: $(10, m)$
- $dZ_{\text{hidden}}$: $(10, m)$
- $X^T$: $(m, 784)$
- $dW_{\text{hidden}}$: $(10, 784)$ ✓

---

# Gradient Descent (Parameter Update)

All parameters are updated simultaneously using standard gradient descent with learning rate $\alpha$:

$$W_{\text{hidden}} \leftarrow W_{\text{hidden}} - \alpha \cdot dW_{\text{hidden}}$$

$$b_{\text{hidden}} \leftarrow b_{\text{hidden}} - \alpha \cdot db_{\text{hidden}}$$

$$W_{\text{output}} \leftarrow W_{\text{output}} - \alpha \cdot dW_{\text{output}}$$

$$b_{\text{output}} \leftarrow b_{\text{output}} - \alpha \cdot db_{\text{output}}$$

Default hyperparameters:
```python
learning_rate = 0.10
epochs        = 500
```

---

# Accuracy

Predictions are taken as the class with the highest probability:

$$\hat{y} = \arg\max_k\ A_{\text{out},k}$$

Accuracy is computed as:

$$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\left[\hat{y}^{(i)} = y^{(i)}\right]$$

---

# Visualizing Predictions

```python
visualize_prediction(0, W_hidden, b_hidden, W_output, b_output)
```

This displays the 28×28 image alongside the model's prediction and the true label.

---

# Expected Results

| Set | Accuracy |
|-----|----------|
| Training | ~85–90% after 500 epochs |
| Validation | ~84–88% |

> Results vary slightly due to random weight initialization.

---

# Key Concepts Summary

| Concept | Formula |
|---------|---------|
| Normalize | $X / 255$ |
| ReLU | $\max(0, Z)$ |
| Softmax | $e^{Z_k} / \sum e^{Z_j}$ |
| Cross-Entropy Loss | $-\frac{1}{m}\sum Y \log(A)$ |
| Output gradient | $A_{\text{out}} - Y_{\text{one-hot}}$ |
| Hidden gradient | $W_{\text{out}}^T \cdot dZ_{\text{out}} \odot \text{ReLU}'(Z)$ |
| Weight update | $W \leftarrow W - \alpha \cdot dW$ |

---

# References

- [3Blue1Brown — Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Samson Zhang — Building a Neural Network from Scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU)
- [Kaggle Digit Recognizer Competition](https://www.kaggle.com/c/digit-recognizer)

---

# License

MIT License — free to use, modify, and distribute.
