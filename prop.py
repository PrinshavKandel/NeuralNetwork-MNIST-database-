def backward_pass(z_hidden, a_hidden, z_out, a_out, W_hidden, W_output, images, labels):
    encoded_labels = encode_labels(labels)
    dz_out = a_out - encoded_labels
    dW_output = 1 / num_train_samples * dz_out.dot(a_hidden.T)
    db_output = 1 / num_train_samples * np.sum(dz_out)
    dz_hidden = W_output.T.dot(dz_out) * relu_derivative(z_hidden)
    dW_hidden = 1 / num_train_samples * dz_hidden.dot(images.T)
    db_hidden = 1 / num_train_samples * np.sum(dz_hidden)
    return dW_hidden, db_hidden, dW_output, db_output

def update_weights(W_hidden, b_hidden, W_output, b_output, dW_hidden, db_hidden, dW_output, db_output, learning_rate):
    W_hidden -= learning_rate * dW_hidden
    b_hidden -= learning_rate * db_hidden
    W_output -= learning_rate * dW_output
    b_output -= learning_rate * db_output
    return W_hidden, b_hidden, W_output, b_output