import numpy as np
import matplotlib.pyplot as plt

'''
Activation Function:
An activation function is a function (for example, ReLU or sigmoid) that takes in the weighted sum of all of the inputs 
from the previous layer and then generates and passes an output value (typically nonlinear) to the next layer.

Rectified Linear Unit (ReLU):
If input is negative or zero, output is 0.
If input is positive, output is equal to input.

Sigmoid Function:
A function that maps logistic or multinomial regression output (log odds) to probabilities, returning a value between 
0 and 1. The sigmoid function has the following formula:

y = 1 / (1 + e^-sigma) where sigma = logistic regression (value between 0 and 1)

Activation Function tanh:
The range of the tanh function is from (-1 to 1). tanh is also sigmoidal. The advantage is that the negative inputs 
will be mapped strongly negative and the zero inputs will be mapped near zero, see the tanh graph
'''

x = np.arange(-4 * np.pi, 4 * np.pi, 0.01)  # tanh to +-4pi
t = np.tanh(x)
plt.plot(x, t)

plt.title("Using numpy.tanh() for the tanh Function")
plt.legend(['tanh(x)'])
plt.grid()
plt.show()
