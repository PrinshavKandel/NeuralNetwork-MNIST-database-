def get_predictions(a_out):
    return np.argmax(a_out, 0)

def get_accuracy(predictions, labels):
    print(predictions, labels)
    return np.sum(predictions == labels) / labels.size

def train(images, labels, learning_rate, epochs):
    W_hidden, b_hidden, W_output, b_output = init_weights()
    for epoch in range(epochs):
        z_hidden, a_hidden, z_out, a_out = forward_pass(W_hidden, b_hidden, W_output, b_output, images)
        dW_hidden, db_hidden, dW_output, db_output = backward_pass(z_hidden, a_hidden, z_out, a_out, W_hidden, W_output, images, labels)
        W_hidden, b_hidden, W_output, b_output = update_weights(W_hidden, b_hidden, W_output, b_output, dW_hidden, db_hidden, dW_output, db_output, learning_rate)
        if epoch % 10 == 0:
            print("Epoch: ", epoch)
            predictions = get_predictions(a_out)
            print(get_accuracy(predictions, labels))
    return W_hidden, b_hidden, W_output, b_output

W_hidden, b_hidden, W_output, b_output = train(train_images, train_labels, 0.10, 500)

def predict(images, W_hidden, b_hidden, W_output, b_output):
    _, _, _, a_out = forward_pass(W_hidden, b_hidden, W_output, b_output, images)
    predictions = get_predictions(a_out)
    return predictions
