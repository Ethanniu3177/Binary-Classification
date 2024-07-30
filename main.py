# Import packages
import pandas as pd
import numpy as np
import os
import requests

# Remove the files if they exist
if os.path.exists('winequality-red.csv'):
    os.remove('winequality-red.csv')
if os.path.exists('winequality-white.csv'):
    os.remove('winequality-white.csv')


# Function to download a file from a URL
def download_file(url, filename):
    response = requests.get(url, verify=True)
    with open(filename, 'wb') as file:
        file.write(response.content)


# Download the files
download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
              'winequality-red.csv')
download_file('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
              'winequality-white.csv')

# Read the red wine csv and assign them as 1
df_red = pd.read_csv('winequality-red.csv', delimiter=";")
df_red["color"] = 1
# print(df_red.head())

# Read the white wine csv and assign them as 0
df_white = pd.read_csv('winequality-white.csv', delimiter=";")
df_white["color"] = 0
# print(df_white.head())

# We choose three attributes of the wine to perform our prediction on
input_columns = ["citric acid", "residual sugar", "total sulfur dioxide"]
output_columns = ["color"]

# Now we combine our two dataframes
df = pd.concat([df_red, df_white])

# And shuffle them in place to mix the red and white wine data together
df = df.sample(frac=1).reset_index(drop=True)
print(df.head())

# We extract the relevant features into our X and Y numpy arrays
X = df[input_columns].to_numpy()
Y = df[output_columns].to_numpy()
print("Shape of X:", X.shape)
print("Shape of Y:", Y.shape)
# input features
in_features = X.shape[1]


# Function for evaluating classification accuracy
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
    print("Our model predicted", correct, "out of", num_samples,
          "correctly for", accuracy * 100, "% accuracy")
    return accuracy

# Create the classifier
class WineModel:
    def __init__(self, in_features):
        # Better, we set initial weights to small normally distributed values.
        self.w = 0.01 * np.random.randn(in_features)
        self.w_0 = 0.01 * np.random.randn()
        self.non_zero_tolerance = 1e-8 # add this to divisions to ensure we don't divide by 0

    def forward(self, x):
        # Calculate and save the pre-activation z
        self.z = x @ self.w.T + self.w_0

        # Apply the activation function, and return
        self.a = self.activation(self.z)
        return self.a

    # update weights based on gradients and learning rate
    def update(self, grad_loss, learning_rate):
        self.w   -= grad_loss * self.grad_w   * learning_rate
        self.w_0 -= grad_loss * self.grad_w_0 * learning_rate

# New implementation! Single neuron classification model
class WineClassificationModel(WineModel):
    # Sigmoid activation function for classification
    def activation(self, z):
        return 1 / (1 + np.exp(-z) + self.non_zero_tolerance)

    # Gradient of output w.r.t. weights, for sigmoid activation
    def gradient(self, x):
        self.grad_w = self.a * (1-self.a) * x
        self.grad_w_0 = self.a * (1-self.a)


def train_model_NLL_loss(model, input_data, output_data,
                         learning_rate, num_epochs):
    non_zero_tolerance = 1e-8 # add this to the log calculations to ensure we don't take the log of 0
    num_samples = len(input_data)
    for epoch in range(1, num_epochs+1):
        total_loss = 0 #keep track of total loss across the data set

        for i in range(num_samples):
            x = input_data[i,...]
            y = output_data[i]
            y_predicted = model.forward(x)

            # NLL loss function
            loss = -(y * np.log(y_predicted + non_zero_tolerance) + (1-y) * np.log(1-y_predicted + non_zero_tolerance))
            total_loss += loss

            # gradient of prediction w.r.t. weights
            model.gradient(x)

            #gradient of loss w.r.t. prediction, for NLL
            grad_loss = (y_predicted - y)/(y_predicted * (1-y_predicted))

            # update our model based on gradients
            model.update(grad_loss, learning_rate)

        report_every = max(1, num_epochs // 10)
        if epoch == 1 or epoch % report_every == 0: #every few epochs, report
            print("epoch", epoch, "has total loss", total_loss)

# Train the model
learning_rate = 0.001
epochs = 200

classification_model = WineClassificationModel(in_features=len(X[0]))

train_model_NLL_loss(classification_model, X, Y, learning_rate, epochs)
print("\nFinal weights:")
print(classification_model.w, classification_model.w_0)
evaluate_classification_accuracy(classification_model, X, Y)
