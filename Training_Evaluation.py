import numpy as np


def train_model_NLL_loss(model, input_data, output_data, learning_rate, num_epochs):
    non_zero_tolerance = (
        1e-8  # add this to the log calculations to ensure we don't take the log of 0
    )
    num_samples = len(input_data)
    for epoch in range(1, num_epochs + 1):
        total_loss = 0  # keep track of total loss across the data set

        for i in range(num_samples):
            x = input_data[i, ...]
            y = output_data[i]
            y_predicted = model.forward(x)

            # NLL loss function
            loss = -(
                y * np.log(y_predicted + non_zero_tolerance)
                + (1 - y) * np.log(1 - y_predicted + non_zero_tolerance)
            )
            total_loss += loss

            # gradient of prediction w.r.t. weights
            model.gradient(x)

            # gradient of loss w.r.t. prediction, for NLL
            grad_loss = (y_predicted - y) / (y_predicted * (1 - y_predicted))

            # update our model based on gradients
            model.update(grad_loss, learning_rate)

        report_every = max(1, num_epochs // 10)
        if epoch == 1 or epoch % report_every == 0:  # every few epochs, report
            print("epoch", epoch, "has total loss", total_loss)


def evaluate_classification_accuracy(model, input_data, labels):
    # Count the number of correctly classified samples given a set of weights
    correct = 0
    num_samples = len(input_data)
    for i in range(num_samples):
        x = input_data[i, ...]
        y = labels[i]
        y_predicted = model.forward(x)
        label_predicted = 1 if y_predicted > 0.5 else 0
        if label_predicted == y:
            correct += 1
    accuracy = correct / num_samples
    print(
        f"Our model predicted {correct} out of"
        f" {num_samples} correctly for "
        f"{accuracy * 100}% accuracy"
    )
    return accuracy

