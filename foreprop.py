def init_weights():
    W_hidden = np.random.rand(10, 784) - 0.5
    b_hidden = np.random.rand(10, 1) - 0.5
    W_output = np.random.rand(10, 10) - 0.5
    b_output = np.random.rand(10, 1) - 0.5
    return W_hidden, b_hidden, W_output, b_output
def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    return z > 0

def softmax(z):
    activations = np.exp(z) / sum(np.exp(z))
    return activations

def forward_pass(W_hidden, b_hidden, W_output, b_output, images):
    z_hidden = W_hidden.dot(images) + b_hidden
    a_hidden = relu(z_hidden)
    z_out = W_output.dot(a_hidden) + b_output
    a_out = softmax(z_out)
    return z_hidden, a_hidden, z_out, a_out

def encode_labels(labels):
    encoded = np.zeros((labels.size, labels.max() + 1))
    encoded[np.arange(labels.size), labels] = 1
    encoded = encoded.T
    return encoded