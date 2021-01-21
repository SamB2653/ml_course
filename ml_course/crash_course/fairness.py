from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import seaborn as sns

# Display / format options
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
pd.options.display.max_columns = 500
pd.options.display.max_rows = 100

# Loading the data set
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

train_csv = tf.keras.utils.get_file('adult.data',
                                    'https://download.mlcc.google.com/mledu-datasets/adult_census_train.csv')
test_csv = tf.keras.utils.get_file('adult.data',
                                   'https://download.mlcc.google.com/mledu-datasets/adult_census_test.csv')

train_df = pd.read_csv(train_csv, names=COLUMNS, sep=r'\s*,\s*',
                       engine='python', na_values="?")
test_df = pd.read_csv(test_csv, names=COLUMNS, sep=r'\s*,\s*', skiprows=[0],
                      engine='python', na_values="?")

"""
Some important questions to investigate when auditing a dataset for fairness:
* Are there missing feature values for a large number of observations?
* Are there features that are missing that might affect other features?
* Are there any unexpected feature values?
* What signs of data skew do you see?
"""

# Data frame information
# Work class and Occupation have over 5% missing values, due to being low these rows can be dropped
print(train_df.describe())  # general info about data
print(train_df.info())  # data frame info
print(train_df.isnull().sum() / len(train_df) * 100)  # % of NaN data

"""
Data info findings:
hours_per_week has a min of 1 hour which does not seem correct
capital_gain and capital_loss values are very low, implying many 0 values skewing data, however this is just for 
investments so could be correct
"""


# Predicting income using Keras


def pandas_to_numpy(data):
    """Convert a pandas DataFrame into a Numpy array (pandas not accepted input)"""
    # Drop empty rows.
    data = data.dropna(how="any", axis=0)

    # Separate DataFrame into two Numpy arrays"
    labels = np.array(data['income_bracket'] == ">50K")
    features = data.drop('income_bracket', axis=1)
    features = {name: np.array(value) for name, value in features.items()}

    return features, labels


# Create categorical feature columns
# Since we don't know the full range of possible values with occupation and
# native_country, use categorical_column_with_hash_bucket() to help map
# each feature string into an integer ID.
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# For the remaining categorical features, since we know what the possible values
# are, we can be more explicit and use categorical_column_with_vocabulary_list()
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])
race = tf.feature_column.categorical_column_with_vocabulary_list(
    "race", [
        "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
    ])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

# Make Age a Categorical Feature
age = tf.feature_column.numeric_column('age')
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# Define the Model Features
# List of variables, with special handling for gender subgroup.
variables = [native_country, education, occupation, workclass,
             relationship, age_buckets]
subgroup_variables = [gender]
feature_columns = variables + subgroup_variables

# Train a Deep Neural Net Model on Adult Data set

"""
Convert the high-dimensional categorical features into a low-dimensional and dense real-valued vector, an embedding 
vector. indicator_column (think of it as one-hot encoding) and embedding_column (that converts sparse features into 
dense features) helps us streamline the process. Then define a feed-forward neural network with two hidden layers.

Use: workclass, education, age_buckets, relationship, native_country and occupation.
"""

deep_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(age_buckets),
    tf.feature_column.indicator_column(relationship),
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
]

# Define Deep Neural Net Model
# Parameters from form fill-ins
HIDDEN_UNITS_LAYER_01 = 128
HIDDEN_UNITS_LAYER_02 = 64
LEARNING_RATE = 0.1
L1_REGULARIZATION_STRENGTH = 0.001
L2_REGULARIZATION_STRENGTH = 0.001

RANDOM_SEED = 512
tf.random.set_seed(RANDOM_SEED)

# List of built-in metrics that we'll need to evaluate performance.
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'),
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

regularizer = tf.keras.regularizers.l1_l2(
    l1=L1_REGULARIZATION_STRENGTH, l2=L2_REGULARIZATION_STRENGTH)

model = tf.keras.Sequential([
    layers.DenseFeatures(deep_columns),
    layers.Dense(
        HIDDEN_UNITS_LAYER_01, activation='relu', kernel_regularizer=regularizer),
    layers.Dense(
        HIDDEN_UNITS_LAYER_02, activation='relu', kernel_regularizer=regularizer),
    layers.Dense(
        1, activation='sigmoid', kernel_regularizer=regularizer)
])

model.compile(optimizer=tf.keras.optimizers.Adagrad(LEARNING_RATE),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=METRICS)

# Fit Deep Neural Net Model to the Adult Training Dataset, pass through the full training data 10 times.
EPOCHS = 10
BATCH_SIZE = 500
features, labels = pandas_to_numpy(train_df)
model.fit(x=features, y=labels, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Evaluate Deep Neural Net Performance
features, labels = pandas_to_numpy(test_df)
model.evaluate(x=features, y=labels)

"""
Best outcome:
loss: 0.4250 - tp: 4319.0000 - fp: 1906.0000 - tn: 20747.0000 - fn: 3189.0000 - accuracy: 0.8311
precision: 0.6938 - recall: 0.5753 - auc: 0.8815
"""


# Define Function to Visualize Binary Confusion Matrix
def plot_confusion_matrix(
        confusion_matrix, class_names, subgroup, figsize=(8, 6)):
    # We're taking our calculated binary confusion matrix that's already in the
    # form of an array and turning it into a pandas DataFrame because it's a lot
    # easier to work with a pandas DataFrame when visualizing a heat map in
    # Seaborn.
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    sns.set_context("notebook", font_scale=1.25)
    fig = plt.figure(figsize=figsize)
    plt.title('Confusion Matrix for Performance Across ' + subgroup)

    # Combine the instance (numercial value) with its description
    strings = np.asarray([['True Positives', 'False Negatives'],
                          ['False Positives', 'True Negatives']])
    labels = (np.asarray(
        ["{0:g}\n{1}".format(value, string) for string, value in zip(
            strings.flatten(), confusion_matrix.flatten())])).reshape(2, 2)

    heatmap = sns.heatmap(df_cm, annot=labels, fmt="",
                          linewidths=2.0, cmap=sns.color_palette("GnBu_d"));
    heatmap.yaxis.set_ticklabels(
        heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(
        heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('References')
    plt.xlabel('Predictions')
    plt.show()
    return fig


# Visualize Binary Confusion Matrix and Compute Evaluation Metrics Per Subgroup
CATEGORY = "gender"
SUBGROUP = "Female"

# Labels for annotating axes in plot.
classes = ['Over $50K', 'Less than $50K']

# Given define subgroup, generate predictions and obtain its corresponding
# ground truth.
subgroup_filter = test_df.loc[test_df[CATEGORY] == SUBGROUP]
features, labels = pandas_to_numpy(subgroup_filter)
subgroup_results = model.evaluate(x=features, y=labels, verbose=0)
confusion_matrix = np.array([[subgroup_results[1], subgroup_results[4]],
                             [subgroup_results[2], subgroup_results[3]]])

subgroup_performance_metrics = {
    'ACCURACY': subgroup_results[5],
    'PRECISION': subgroup_results[6],
    'RECALL': subgroup_results[7],
    'AUC': subgroup_results[8]
}
performance_df = pd.DataFrame(subgroup_performance_metrics, index=[SUBGROUP])
pd.options.display.float_format = '{:,.4f}'.format

print(performance_df)
plot_confusion_matrix(confusion_matrix, classes, SUBGROUP)


"""
From the confusion matrix (different sub categories tried):
The model performs better for female than male, this suggests that the model is over fitting, particularly for female 
and lower income bracket. The model will not generalize well and there is a disproportionately small number of high 
income bracket compared to low income bracket for males.
"""
