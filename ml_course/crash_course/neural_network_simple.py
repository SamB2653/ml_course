from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
print("Imported modules.")

# Load the data set
train_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")
train_df = train_df.reindex(np.random.permutation(train_df.index))  # shuffle the examples
test_df = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv")

# Normalize values
train_df_mean = train_df.mean()
train_df_std = train_df.std()
train_df_norm = (train_df - train_df_mean) / train_df_std  # Calculate the Z-scores of each column in the training set

test_df_mean = test_df.mean()
test_df_std = test_df.std()
test_df_norm = (test_df - test_df_mean) / test_df_std  # Calculate the Z-scores of each column in the test set
print("Normalized the values.")

# Represent data (feature layer with 3 features: latitude X longitude (a feature cross), median_income and population)
feature_columns = []  # Create an empty list that holds all feature columns
resolution_in_Zs = 0.3  # 3/10 of a standard deviation. Due to Z scaling on all columns Z. 1 Z = 1 standard deviation

# Create a bucket feature column for latitude:
latitude_as_a_numeric_column = tf.feature_column.numeric_column("latitude")  # Select column to be used, numeric values
latitude_boundaries = list(np.arange(int(min(train_df_norm['latitude'])),
                                     int(max(train_df_norm['latitude'])),
                                     resolution_in_Zs))  # Min, max and step
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)
# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df_norm['longitude'])),
                                      int(max(train_df_norm['longitude'])),
                                      resolution_in_Zs))
longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)
# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Represent median_income as a floating-point value.
median_income = tf.feature_column.numeric_column("median_income")
feature_columns.append(median_income)

# Represent population as a floating-point value.
population = tf.feature_column.numeric_column("population")
feature_columns.append(population)

# Convert the list of feature columns into a layer that will later be fed into the model
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


def plot_the_loss_curve(epochs, mse):  # Graphing
    """Plot a curve of loss vs. epoch."""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.03])
    plt.show()


''' 
# Build a linear regression model as a baseline
def create_model(my_learning_rate, feature_layer):
    """Create and compile a simple linear regression model."""
    model = tf.keras.models.Sequential()  # Most simple tf.keras models are sequential.
    model.add(feature_layer)  # Add the layer containing the feature columns to the model.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))  # Add one linear layer to the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),  # Construct the layers into a model
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """Feed a dataset into the model in order to train it."""
    # Split the dataset into features and label.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True, verbose=2)  # verbose=2 to clean up console output
    # Get details that will be useful for plotting the loss curve.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    rmse = hist["mean_squared_error"]
    return epochs, rmse


# The following variables are the hyperparameters.
learning_rate = 0.01
epochs = 15
batch_size = 1000
label_name = "median_house_value"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set.
print("Train linear model")
epochs, mse = train_model(my_model, train_df_norm, epochs, batch_size, label_name)
plot_the_loss_curve(epochs, mse)

test_features = {name: np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name))  # isolate the label
print("\n Evaluate the linear regression model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
'''  # Baseline with a linear regression model


# Define a deep neural net model, create_model defines the number of layers and nodes in each layer (topography)
def create_model(my_learning_rate, my_feature_layer):
    """Create and compile a simple linear regression model."""
    model = tf.keras.models.Sequential()  # Most simple tf.keras models are sequential.
    model.add(my_feature_layer)  # Add the layer containing the feature columns to the model.

    '''
    Describe the topography of the model by calling the tf.keras.layers.Dense
    method once for each layer. We've specified the following arguments:
      * units specifies the number of nodes in this layer.
      * activation specifies the activation function (Rectified Linear Unit - ReLU).
      * name is just a string that can be useful when debugging.
    '''

    # Define the first hidden layer with 20 nodes.
    model.add(tf.keras.layers.Dense(units=20,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.04),
                                    name='Hidden1'))

    # Define the second hidden layer with 12 nodes.
    model.add(tf.keras.layers.Dense(units=12,
                                    activation='relu',
                                    kernel_regularizer=tf.keras.regularizers.l2(l=0.04),
                                    name='Hidden2'))

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


# Define a training function
def train_model(model, dataset, epochs, label_name, batch_size=None):
    """Train the model by feeding it data."""
    features = {name: np.array(value) for name, value in dataset.items()}  # Split the data set into features and label
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)
    epochs = history.epoch  # The list of epochs is stored separately from the rest of history
    hist = pd.DataFrame(history.history)  # gather a snapshot of the model'smse at each epoch to track training
    mse = hist["mean_squared_error"]
    return epochs, mse


# Call the functions to build and train a deep neural net
# The following variables are the hyperparameters.
learning_rate = 0.007
epochs = 140
batch_size = 1000

# Specify the label
label_name = "median_house_value"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the normalized training set. We're passing the entire normalized training set, but the model will
# only use the features defined by the feature_layer.
epochs, mse = train_model(my_model, train_df_norm, epochs, label_name, batch_size)
plot_the_loss_curve(epochs, mse)

# After building a model against the training set, test that model against the test set.
test_features = {name: np.array(value) for name, value in test_df_norm.items()}
test_label = np.array(test_features.pop(label_name))  # isolate the label
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)
