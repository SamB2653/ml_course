import numpy as np
import pandas as pd

'''
Following the Pandas UltraQuick Tutorial but with some minor changes and additions. Source:
https://colab.research.google.com/github/google/eng-edu/blob/master/ml/cc/exercises/pandas_dataframe_ultraquick_tutorial.ipynb
'''

# Create and populate a 5x2 NumPy array.
my_data = np.array([[0, 3], [10, 7], [20, 9], [30, 14], [40, 15]])
print("Data for DataFrame:\n %s \n" % my_data)

# Create a Python list that holds the names of the two columns.
my_column_names = ['Temperature', 'Activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)
print("Created DataFrame from array values with columns Temperature and Activity:\n %s \n" % my_dataframe)

# Create a new column named Adjusted.
my_dataframe["Adjusted"] = my_dataframe["Activity"] + 2
print("New column \"Adjusted\" added:\n %s \n" % my_dataframe)

# Specifying a subset of a DataFrame - using .head()
print("Rows #0, #1, and #2 using head:\n %s \n" % my_dataframe.head(3))

# Specifying a subset of a DataFrame - using .iloc()
print("Row #2 using iloc:\n %s \n" % my_dataframe.iloc[[2]])

# Specifying a subset of a DataFrame - using [x:y]
print("Rows #1, #2, and #3 using DataFrame[x:y]:\n {0} \n".format(my_dataframe[1:4]))

# Specifying a subset of a DataFrame - using ["ColumnName"]
print("Column Temperature using column name:\n %s \n" % my_dataframe["Temperature"])

'''
Task 1: Create a DataFrame
Create an 3x4 (3 rows x 4 columns) pandas DataFrame in which the columns are named Eleanor, Chidi, Tahani, and Jason. 
Populate each of the 12 cells in the DataFrame with a random integer between 0 and 100, inclusive.

Output the following:
the entire DataFrame
the value in the cell of row #1 of the Eleanor column
Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason.
'''

random_integers = np.random.randint(0, 101, (3, 4))  # Creates 3x4 numpy array with random integers
column_names = ["Eleanor", "Chidi", "Tahani", "Jason"]  # Select column names
created_dataframe = pd.DataFrame(data=random_integers, columns=column_names)  # Create the DataFrame
print("Task 1 DataFrame:\n %s \n" % created_dataframe)  # The entire DataFrame

print("Column Eleanor, Row 1: %d \n" % created_dataframe["Eleanor"].iloc[1])  # Value of row #1 of the Eleanor column

# Create a fifth column named Janet, which is populated with the row-by-row sums of Tahani and Jason
created_dataframe["Janet"] = created_dataframe["Tahani"] + created_dataframe["Jason"]
print("Janet Column = Tahani + Jason:\n %s \n" % created_dataframe)

'''
Referencing: If you assign a DataFrame to a new variable, any change to the DataFrame or to the new variable will be 
reflected in the other.
Copying: If you call the pd.DataFrame.copy method, you create a true independent copy. Changes to the original 
DataFrame or to the copy will not be reflected in the other.
'''

# Create a reference by assigning created_dataframe to a new variable reference_dataframe
reference_dataframe = created_dataframe
print("Experiment with reference and copy:\n %s \n" % reference_dataframe)

# Print the starting value of a particular cell.
print("Reference:\n  Starting value of created_dataframe: %d" % created_dataframe['Jason'][1])
print("  Starting value of reference_dataframe: %d\n" % reference_dataframe['Jason'][1])

# Modify a cell in df. The reference is updated as well when the original DataFrame is modified
created_dataframe.at[1, 'Jason'] = created_dataframe['Jason'][1] + 5
print("  Updated created_dataframe: %d" % created_dataframe['Jason'][1])
print("  Updated reference_dataframe: %d\n\n" % reference_dataframe['Jason'][1])

# Create a true copy of created_dataframe
copy_of_dataframe = created_dataframe.copy()

# Print the starting value of a particular cell.
print("Copy:\n  Starting value of created_dataframe: %d" % created_dataframe['Eleanor'][1])
print("  Starting value of copy_of_dataframe: %d \n" % copy_of_dataframe['Eleanor'][1])

# Modify a cell in created_dataframe. The copy is not updated when then original DataFrame is modified
created_dataframe.at[1, 'Eleanor'] = created_dataframe['Eleanor'][1] + 3
print("  Updated my_dataframe: %d" % created_dataframe['Eleanor'][1])
print("  copy_of_my_dataframe does not get updated: %d" % copy_of_dataframe['Eleanor'][1])
