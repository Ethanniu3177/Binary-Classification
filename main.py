# Import packages
import pandas as pd
import os
import requests
from Classifiers import WineClassificationModel
import Training_Evaluation as te

# Remove the files if they exist
if os.path.exists("winequality-red.csv"):
    os.remove("winequality-red.csv")
if os.path.exists("winequality-white.csv"):
    os.remove("winequality-white.csv")


# Function to download a file from a URL
def download_file(url, filename):
    response = requests.get(url, verify=True)
    with open(filename, "wb") as file:
        file.write(response.content)


# Download the files
download_file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "winequality-red.csv",
)
download_file(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
    "winequality-white.csv",
)

# Read the red wine csv and assign them as 1
df_red = pd.read_csv("winequality-red.csv", delimiter=";")
df_red["color"] = 1
# print(df_red.head())

# Read the white wine csv and assign them as 0
df_white = pd.read_csv("winequality-white.csv", delimiter=";")
df_white["color"] = 0
print(df_white.head())

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
# print("Shape of X:", X.shape)
# print("Shape of Y:", Y.shape)

# Train the model
Learning_rate = 0.001
Epochs = 200

classification_model = WineClassificationModel(in_features=len(X[0]))

te.train_model_NLL_loss(classification_model, X, Y, Learning_rate, Epochs)
print("\nFinal weights:")
print(classification_model.w, classification_model.w_0)
te.evaluate_classification_accuracy(classification_model, X, Y)
