from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Pandas / NumPy display settings
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
np.set_printoptions(linewidth=200)  # numpy array format

# Load the data set (load train/test and split each set into features/labels), .mnist built in
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Data Example
# print(x_train[2900])  # Output a digit
# plt.imshow(x_train[2900])  # Display the digit with false colours
# plt.show()

#  Normalize feature values
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
print(f"Standard:{x_train[2900][6]}")  # Output a row
print(f"Normalized:{x_train_normalized[2900][6]}")  # Output a normalized row


def plot_curve(epochs, hist, list_of_metrics):  # Accuracy curve plot
    """Plot a curve of one or more classification metrics vs epoch. list_of_metrics should be one of the names shown
     in: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics"""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()


def create_model(my_learning_rate):
    """Create and compile a deep neural net."""

    model = tf.keras.models.Sequential()

    # The features are stored in a two-dimensional 28X28 array.
    # Flatten that two-dimensional array into a a one-dimensional
    # 784-element array.
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Define the first hidden layer.
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))

    # Define the second hidden layer.
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))

    # Define a dropout regularization layer.
    model.add(tf.keras.layers.Dropout(rate=0.3))

    # Define the output layer. The units parameter is set to 10 because
    # the model must choose among 10 possible output values (representing
    # the digits from 0 to 9, inclusive).
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Construct the layers into a model that TensorFlow can execute.
    # The loss function for multi-class classification is different than the
    # loss function for binary classification.
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model


def train_model(model, train_features, train_label, epochs,
                batch_size=None, validation_split=0.1):
    """Train the model by feeding it data."""

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True,
                        validation_split=validation_split)

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch.
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


# Hyperparameters.
learning_rate = 0.003
epochs = 50
batch_size = 4000
validation_split = 0.2

# Model's topography.
my_model = create_model(learning_rate)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, x_train_normalized, y_train,
                           epochs, batch_size, validation_split)

# Plot a graph of the metric vs epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=x_test_normalized, y=y_test, batch_size=batch_size)  # loss: 0.0821 - accuracy: 0.9810
