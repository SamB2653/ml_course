# Reducing Loss:  
#### Iterative Learning:
Iteratively reducing loss has the main goal of trying to find the best possible model as efficiently as possible. You 
start with a guess (of weight value) and let the system tell you what the loss is, then take another value (the step is 
determined by the learning rate) and see what the loss is for the next guess, this should be less loss than then 
previous guess. This is continued until the local minimum has been reached and loss cannot decrease any more.  

For example is we use a model that takes one feature and returns one prediction (y<sup>'</sup>)
* y<sup>'</sup> = b + w<sub>1</sub>x<sub>1</sub>  

Initial values need to be chosen for b and w<sub>1</sub>, for this example they are set to 0 but should be random. If
the first feature value is 10 then we can plug this into the formula, we get:

* y<sup>'</sup> = 0 + 0 * 10 = 0

The model can use multiple different kinds of loss functions. If the squared loss function is used then the loss
function takes the inputs y and y<sup>'</sup> (correct label corresponding to features x and the models prediction 
for features x)  

After each step new parameters will be computed as the ML system examines the value of the loss function and will
generate new values of b and w<sub>1</sub>. Then the ML system re-evaluates all those features against all those labels,
yielding a new value for the loss function, which yields new parameter values. This continues iterating until the 
algorithm discovers the model parameters with the lowest possible loss. Usually the process will stop when the change
in loss is very small.

#### Gradient Descent:

![alt text](https://developers.google.com/machine-learning/crash-course/images/convex.svg "weight vs loss graph")

The first step is to pick a starting value for w<sub>1</sub>, most algorithms will pick a value of 0 or pick a random
value. The gradient descent algorithm calculates the gradient of the loss curve at the starting point, the gradient of
loss is equal to the derivative of the slope of the curve. When there are multiple weights, the gradient is a vector of 
partial derivatives with respect to the weights. The gradient will always point in the direction of steepest increase
in the loss function, therefore the gradient descent algorithm will take a step in the direction of the negative 
gradient, in order to reduce loss as quickly as possible.  

To determine the next point on the loss curve, the gradient descent algorithm adds dome fraction of the gradient's 
magnitude to the starting point. The algorithm then repeats this process until the loss only decreases by a very small
amount.

#### Learning Rate:  
Gradient descent algorithms multiply the gradient by a scalar known as the learning rate (also known as the step size)
to determine the next point.  
**Example:** if the gradient magnitude is 2.5 and the learning rate is 0.01 then the next point will be 0.025 away
from the previous point (2.5*0.01).  

#### Hyperparameters:  
These are settings that can be tweaked in ML algorithms, the learning rate is one of these parameters. If a learning 
rate is too small then the learning process will take too long. A learning rate that is too large will never get to the
minimum loss level as it will keep overshooting as the steps are too large. A learning rate that can reach the minimum 
loss in the least amount of steps is the most desirable.

#### Stochastic Gradient Descent (SGD):  
In gradient descent, a batch is the total number of examples you use to calculate the gradient in a single iteration.
A large batch can take a single iteration to take a very long time to compute. A large set of data with random samples
most likely contains redundant data so as the batch size increases the probability of redundant data increases.
By choosing examples at random from our data set, we could estimate (noisily) a big average from a much smaller one.  

SGD uses a single example (batch size of 1) per iteration. Over many iterations SGD will work but have a lot of noise.
Stochastic means that each sample is chosen at random.  

#### Mini-Batch Stochastic Gradient Descent (Mini-Batch SGD):  
Mini-Batch SGD is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 
examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than 
full-batch.
