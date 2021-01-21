#Logistic Regression:  
Instead of predicting exactly 0 or 1, logistic regression generates a probability (value between 0 and 1). So if the 
model infers a value of 0.87 then there is an 87% probability of that outcome. More precisely, it means that in the 
limit of infinite training examples the model predicts 87%.

Many problems require a probability estimate as output. Logistic regression is an extremely efficient mechanism for
calculating probabilities. Practically, you can use the returned probability in either of the following two ways:
* "as is"
* converted to a binary category

####Considering Probability "as is":
Consider a logistic regression model to predict the probability of if it will be sunny or rainy on a set day. The
formula would be:
* _p(sunny|rainy)_

if the logistic regression model predicts a value of 0.25, then over a year it will be sunny for 91 days in a year 
(0.25 * 365).

####Considering Probability "converted to a binary category":
In many cases, you'll map the logistic regression output into the solution to a binary classification problem, 
in which the goal is to correctly predict one of two possible labels (e.g., "spam" or "not spam"). This is demonstrated
in Classification.md.

####Sigmoid Function:
The output always falls between 0 and 1 due to the sigmoid function:

<img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\frac{1}{e^{-z}}" title="y = \frac{1}{e^{-z}}" />  

Z represents the output of the linear layer of the model trained with logistic regression, so sigmoid(z) yields a value
between 0 and 1. It can also be referred to as log-odds because the inverse of the sigmoid states that z can be defined 
as the log of the probability of the "1" label (e.g., "is sunny") divided by the probability of the "0" label
 (e.g., "is rainy"):  
* z = b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>N</sub>x<sub>N</sub>
* w = weights, b = bias and x values are feature values for the point

Which can be graphically displayed as:

![alt text](https://developers.google.com/machine-learning/crash-course/images/SigmoidFunction.png
"Sigmoid")

####Loss Function for Logistic Regression:
The loss function for linear regression is squared loss. The loss function for logistic regression is Log Loss, 
which is defined as follows:  

<img src="https://latex.codecogs.com/gif.latex?Log&space;Loss&space;=&space;\sum_{(x,y)\in&space;D}^{}&space;-&space;ylog(y')&space;-&space;(1&space;-&space;y)log(1-y')" title="Log Loss = \sum_{(x,y)\in D}^{} - ylog(y') - (1 - y)log(1-y')" />  

Where (x,y)D is the data set containing labeled examples, where (x,y) are pairs, y = label in labeled example, y must be
wither 0 or 1. y' = predicted value, somewhere between 0 and 1, given the set of features in x.

####Regularization in Logistic Regression:
Regularization is extremely important in logistic regression modeling. Without regularization, the asymptotic nature
of logistic regression would keep driving loss towards 0 in high dimensions. Consequently, most logistic regression
models use one of the following two strategies to dampen model complexity:
* L<sub>2</sub> regularization.
* Early stopping, that is, limiting the number of training steps or the learning rate.

There is also  L<sub>1</sub> regularization.  

If you assign a unique id to each example, and map each id to its own feature. If you don't specify a regularization
function, the model will become completely overfit. That's because the model would try to drive loss to zero on all
examples and never get there, driving the weights for each indicator feature to +infinity or -infinity. This 
can happen in high dimensional data with feature crosses, when there’s a huge mass of rare crosses that happen
only on one example each.
