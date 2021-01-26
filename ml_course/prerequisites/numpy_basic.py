import numpy as np

'''
Following the NumPy UltraQuick Tutorial but with some minor changes and additions. Source:
https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/numpy_ultraquick_tutorial.ipynb
'''

# 1D array (8 element vector)
one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1, 7.2, 8.3, 9.5])
print("1D Array:\n %s" % one_dimensional_array)

# 2D array (3x2 matrix)
two_dimensional_array = np.array([[6, 5], [11, 7], [4, 8]])
print("2D Array (3x2):\n %s" % two_dimensional_array)

# 3D array (2x2x2 matrix) with all zeroes (np.ones for all ones)
all_zeroes = np.zeros([2, 2, 2])
print("3D Array (2x2x2) - all zeroes:\n %s\n" % all_zeroes)

# Sequence of integers, lower bound (5) included but not the upper bound (12).
sequence_of_integers = np.arange(5, 12)
print("Sequence of integers: %s" % sequence_of_integers)

# Populate an array with random numbers, generates a 6 element vector with random integers between 50 and 100
random_integers_between_50_and_100 = np.random.randint(low=50, high=101, size=6)
print("Random Integers (50-100, 6 elements): %s" % random_integers_between_50_and_100)

# Random float values between 0.0 and 1.0 (5 elements)
random_floats_between_0_and_1 = np.random.random([5])
print("Random floats between 0.0 and 1.0 (original):     %s" % random_floats_between_0_and_1)

# Using broadcasting add 2.0 to the value of every item in the vector created in random_floats_between_0_and_1
random_floats_between_2_and_3 = random_floats_between_0_and_1 + 2.0
print("Random floats between 2.0 and 3.0 (+2 operation): %s" % random_floats_between_2_and_3)

# Using broadcasting multiply by 3 to the value of every item in the vector created in random_floats_between_0_and_1
random_floats_between_6_and_9 = random_floats_between_2_and_3 * 3
print("Random floats between 6.0 and 9.0 (*3 operation): %s" % random_floats_between_6_and_9)

# Task 1: Create a Linear Data set
feature = np.arange(6, 21)  # Assign a sequence of integers from 6 to 20 (inclusive) to a NumPy array named feature
print("\nTask 1:\nSeries of integers between 6 and 20: %s" % feature)
label = (feature * 3) + 4  # Assign 15 values to a NumPy array named label such that: label = (3)(feature) + 4
print("Applying (3)(feature) + 4 operation: %s" % label)

# Task 2: Add Some Noise to the Data set
noise = (np.random.random([15]) * 4) - 2  # Create a noise array having the same dimension as label, no broadcasting
print("\nTask 2:\nAdd Some Noise to the Data set (noise matrix):\n %s" % noise)
label = label + noise  # modify each value assigned to label by adding a different random float value between -2 and +2
print("Applying noise to label:\n %s" % label)
