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
