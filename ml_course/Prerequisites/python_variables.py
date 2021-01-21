"""
Source: https://developers.google.com/machine-learning/crash-course/prereqs-and-prework
"""

# Number Variables (all integers in python3 are long integers)
x = 10  # integer
y = 10.5  # float
z = 3.14J  # complex
print("Number Variables:\n int: %d\n" % x, "float: %d\n" % y, "complex: %s\n" % z)

# String Variables
my_string = "Hello World"  # String
print("String Variables and Simple Operations:")
print(my_string[0])  # 1st character of my_string
print(my_string[1:4])  # 2nd to 4th characters of my_string
print(my_string[4:])  # from 5th character onwards
print(my_string * 2)  # prints my_string 2 times
print(my_string + "Concatenate")  # concatenates strings

# Python Lists - can be modified []
print("\nPython Lists:")
my_list = ["a", "b", "c"]  # create a list of strings
my_list_int = [1, 2, 3]  # create a list of integers
print("List Example: %s" % my_list)
concat_list = my_list + my_list_int
print("Concat a list: %s" % concat_list)
print("The same operations can be completed on a list as used in the string section")

# Python Tuples - tuples are read only lists ()
print("\nPython Tuples:")
my_tuple = ("d", "e", "f")
my_tuple_int = (4, 5, 6)
print("Tuples Example: {0}".format(my_tuple))
concat_tuple = my_tuple + my_tuple_int
print("Concat a tuple: {0}".format(concat_tuple))

# Python Dictionary
print("\nPython Dictionary:")
my_dict = {'letter': 'a', 'number': 20, 'another_string': 'String'}
print("Dictionary Example: {0}".format(my_dict))
print("Dictionary Keys: {0}".format(my_dict.keys()))
print("Dictionary Values: {0}".format(my_dict.values()))

'''
Terms, Factors and Coefficients:
Example: (2 * 3) + 4 - 15x

Terms: The first term is (2 * 3), the second term is 4 and the third term is 15x  
Factors: The factors of the first term are 2 and 3, the second term has one factor 4, and the third term has the 
factors 15 and x
Coefficients: For 15x the value 15 is the coefficient
'''





