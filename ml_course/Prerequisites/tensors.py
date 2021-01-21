import tensorflow as tf
import numpy as np

'''
Following an Introduction to Tensors but with some minor changes and additions. 
Source: https://www.tensorflow.org/guide/tensor
'''

# Create a scalar (rank-0) tensor. A scalar has no axes and is a single value. This will be an int32 tensor by default
rank_0_tensor = tf.constant(4)
print("Rank-0 Tensor (scalar):\n %s" % rank_0_tensor)

# Create a vector (rank-1) tensor. Vector tensors are like lists of values. A vector has 1-axis.
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])  # creating a float tensor
print("Rank-1 Tensor (vector):\n %s" % rank_1_tensor)

# Create a matrix (rank-2) tensor. Has 2-axis
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)  # dtype can be set at creation
print("Rank-2 Tensor (matrix):\n %s" % rank_2_tensor)

# Scalar shape = [], Vector shape = [3], Matrix shape = [3,2]
print("Rank-0 Shape: %s" % rank_0_tensor.shape)
print("Rank-1 Shape: %s" % rank_1_tensor.shape)
print("Rank-2 Shape: %s" % rank_2_tensor.shape)


# There can be an arbitrary number of axes (sometimes called "dimensions"), for example a tensor with 3-axis:
rank_3_tensor = tf.constant([
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9]],
    [[10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]],
    [[20, 21, 22, 23, 24],
     [25, 26, 27, 28, 29]], ])

print("\nRank-3 Tensor (3 dimensions):\n %s" % rank_3_tensor)

# You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:
numpy_method_2d = np.array(rank_2_tensor)  # tensor to numpy array via numpy method
tensor_method_2d = rank_2_tensor.numpy()  # tensor to numpy array via tensor method

# We can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication
print("\nBasic Mathematical Operations on Tensors:")
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])  # could have also said tf.ones([2,2])

print("Element-wise addition:\n %s" % tf.add(a, b), "\n")  # element-wise addition
print("Element-wise multiplication:\n %s" % tf.multiply(a, b), "\n")  # element-wise multiplication
print("Matrix multiplication:\n %s" % tf.matmul(a, b), "\n")  # matrix multiplication

# Tensors are used in all kinds of operations (ops)
c = tf.constant([[4.0, 5.0], [10.0, 1.0]])
print("Tensors are used in various types of operations:\n %s\n" % c)
print("Find the largest value:\n %s\n" % tf.reduce_max(c))  # Find the largest value
print("Find the index of the largest value:\n %s\n" % tf.argmax(c))  # Find the index of the largest value
print("Compute the softmax:\n %s\n" % tf.nn.softmax(c))  # Compute the softmax

'''
Shape: The length (number of elements) of each of the dimensions of a tensor.
Rank: Number of tensor dimensions. A scalar has rank 0, a vector has rank 1, a matrix is rank 2.
Axis or Dimension: A particular dimension of a tensor.
Size: The total number of items in the tensor, the product shape vector
'''

# A rank 4 example - batch is first followed by spatial dimensions and then features for each location last
rank_4_tensor = tf.zeros([3, 2, 4, 5])  # Batch (3), Width (2) , Height (4), Features (5)
print("Rank-4 Example:\n %s\n" % rank_4_tensor)
print("Type of every element:", rank_4_tensor.dtype)
print("Number of dimensions:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# Indexing - TensorFlow follows standard python indexing rules; indexes start at 0 , [-1] and start:stop:step
rank_1_tensor = tf.constant([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
print("\nIndexing Example with a Rank-1 Tensor:\n %s" % rank_1_tensor.numpy())
print("\tFirst:", rank_1_tensor[0].numpy())  # Indexing with a scalar removes the dimension ([0],[1],[-1], ect...)
print("\tSecond:", rank_1_tensor[1].numpy())
print("\tLast:", rank_1_tensor[-1].numpy())

print("\tEverything:", rank_1_tensor[:].numpy())  # Indexing with a : slice keeps the dimension:
print("\tBefore 4:", rank_1_tensor[:4].numpy())
print("\tFrom 4 to the end:", rank_1_tensor[4:].numpy())
print("\tFrom 2, before 7:", rank_1_tensor[2:7].numpy())
print("\tEvery other item:", rank_1_tensor[::2].numpy())
print("\tReversed:", rank_1_tensor[::-1].numpy())

# Multi-axis indexing. Higher rank tensors are indexed by passing multiple indices.
print("\nMulti-Axis-Indexing Example with a Rank-2 Tensor:\n %s" % rank_2_tensor.numpy())
print("\tSingle value: %s" % rank_2_tensor[1, 1].numpy())  # Pull out a single value from a 2-rank tensor
print("\tSecond row:", rank_2_tensor[1, :].numpy())  # get row and column tensors
print("\tSecond column:", rank_2_tensor[:, 1].numpy())
print("\tLast row:", rank_2_tensor[-1, :].numpy())
print("\tFirst item in last column:", rank_2_tensor[0, -1].numpy())
print("\tSkip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# Manipulating Shapes, The tf.reshape operation is fast and cheap as the underlying data does not need to be duplicated
var_x = tf.Variable(tf.constant([[1], [2], [3]]))
print("Manipulating the Shape of Tensors:\n %s" % var_x)
print("\tTensorShape: %s" % var_x.shape)  # Shape returns a "TensorShape" object that shows the size on each dimension
reshaped = tf.reshape(var_x, [1, 3])  # We can reshape a tensor to a new shape (passing in list).
print("\tReshaped: %s\n" % reshaped.shape)  # reshaped

# If you flatten a tensor you can see what order it is laid out in memory.
print("Rank-3 Tensor:\n %s" % rank_3_tensor)
print("Flattened:\n %s" % tf.reshape(rank_3_tensor, [-1]))  # A "-1" passed in the "shape" argument says "Whatever fits"
# Typically the only reasonable uses of tf.reshape are to combine or split adjacent axes (or add/remove 1s).
print("\n3x2x5 to (3x2)x5:\n %s" % tf.reshape(rank_3_tensor, [3*2, 5]))  # 3x2x5 tensor, reshaping to (3x2)x5
print("\n3x2x5 to 3x(2x5):\n %s" % tf.reshape(rank_3_tensor, [3, -1]))  # 3x2x5 tensor, reshaping to 3x(2x5)

# Change Tensors dtype. TensorFlow converts Python integers to tf.int32 and python floating point numbers to tf.float32
the_f64_tensor = tf.constant([2.2, 3.3, 4.4], dtype=tf.float64)
print("Changing Tensors dtype:\n %s" % the_f64_tensor)
the_f16_tensor = tf.cast(the_f64_tensor, dtype=tf.float16)
print("f16_tensor:\n %s" % the_f16_tensor)
the_u8_tensor = tf.cast(the_f16_tensor, dtype=tf.uint8)  # cast to an uint8 and lose the decimal precision
print("u8_tensor:\n %s\n" % the_u8_tensor)

'''
Broadcasting:

Broadcasting is a concept borrowed from the equivalent feature in NumPy. In short, under certain conditions, smaller 
tensors are "stretched" automatically to fit larger tensors when running combined operations on them.
The simplest and most common case is when you attempt to multiply or add a tensor to a scalar. In that case, the scalar 
is broadcast to be the same shape as the other argument.

Example:
A broadcasted add: a [3, 1] times a [1, 4] gives a [3,4]
'''

print("Broadcasting:")  # 1D are stretched to match the other arguments
x = tf.constant([1, 2, 3])
y = tf.constant(2)
z = tf.constant([2, 2, 2])
# All of these are the same computation
print(tf.multiply(x, 2))  # [1 2 3] * [2 2 2] = [2 4 6]
print(x * y)  # [1 2 3] * [2 2 2] = [2 4 6]
print(x * z)  # [1 2 3] * [2 2 2] = [2 4 6]
# Can also used broadcast_to for a demonstration
print("broadcast_to from [1, 2, 3] to [3, 3] shape:\n %s\n" % tf.broadcast_to(tf.constant([1, 2, 3]), [3, 3]))

# Ragged Tensors - A tensor with variable numbers of elements along some axis is called "ragged"
print("Ragged Tensors: The shape of a tf.RaggedTensor contains unknown dimensions:")
ragged_list = [
    [0, 1, 2, 3],
    [4, 5],
    [6, 7, 8],
    [9]]

ragged_tensor = tf.ragged.constant(ragged_list)
print(ragged_tensor)
print(ragged_tensor.shape)

# String Tensors - represent data as strings (variable-length byte arrays) in tensors, length of string not a dimension
print("\nString Tensors - represent data as strings:")
scalar_string_tensor = tf.constant("Gray wolf")  # scalar string.
print("Scalar String", scalar_string_tensor)
# We can use split to split a string into a set of tensors
print("Split a string into a set of tensors: ", tf.strings.split(scalar_string_tensor, sep=" "))

# If we have three string tensors of different lengths, this is OK.
tensor_of_strings = tf.constant(["Gray wolf",  # vector of Strings
                                 "Quick brown fox",
                                 "Lazy dog"])
# Note that the shape is (3,). The string length is not included.
print("Vector of Strings:", tensor_of_strings)  # b prefix = tf.string dtype is not a unicode string, but a byte-string
# Splitting Tensor of Strings, each string might be split into a different number of parts so becomes a RaggedTensor
print("Split a Tensor of Strings (RaggedTensor): ", tf.strings.split(tensor_of_strings, sep=" "))


# Strings can be set to numbers using tf.string.to_number
text = tf.constant("1 10 100")
print("\nStrings can be set to numbers, \"1 10 100\": ", tf.strings.to_number(tf.strings.split(text, " ")))

# Convert string tensor to bytes and then numbers
print("\nConvert string tensor to bytes and then numbers:")
byte_strings = tf.strings.bytes_split(tf.constant("Duck"))
byte_ints = tf.io.decode_raw(tf.constant("Duck"), tf.uint8)
print("Byte strings:", byte_strings)
print("Bytes:", byte_ints)

# Or split it up as unicode and then decode it
print("\nOr split it up as unicode and then decode it:")
unicode_bytes = tf.constant("アヒル 🦆")
unicode_char_bytes = tf.strings.unicode_split(unicode_bytes, "UTF-8")
unicode_values = tf.strings.unicode_decode(unicode_bytes, "UTF-8")
print("Unicode bytes:", unicode_bytes)
print("Unicode chars:", unicode_char_bytes)
print("Unicode values:", unicode_values)

# Sparse Tensors - store values by index in a memory-efficient manner
print("\nSparse Tensors:")
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor)
# We can convert sparse tensors to dense
print("Sparse Tensor to Dense Tensor:\n", tf.sparse.to_dense(sparse_tensor))
