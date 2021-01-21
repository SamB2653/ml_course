#Key Terminology:  
####Labels:
A label is the thing that we are predicting, this can be denoted as **y**.  
**Example:** a label could be something 
such as the future price of an item or what kind of animal a picture is ect...   
####Features:
A feature is the input variable, this can be denoted as **x** or (**x<sub>1</sub>**, **x<sub>2</sub>**,...**x<sub>N
</sub>**) for more complex projects.  
**Example:** for a spam detector the features could include words in the email text, the sender's address, the time the 
email was sent ect...  
##Examples:
An example is a particular instance of data **x** (vector), there are labeled and unlabeled examples.
####Labeled Example:
Includes both features and the label, so:  
* {features, label}: (x, y)

Labeled examples are used to **train** the model.  
**Example:**  for the spam filter example the labeled examples would be individual emails that users have explicitly
marked "spam" or "not spam".
####Unlabeled Example:
Includes features but not the label, so:  
* {features, ?}: (x, ?)

Unlabeled examples are used for the model to predict the label from the features that have been provided. This will be
done after the model has been trained using the labeled examples.  
**Example:** for the spam filter example unlabeled examples are the new emails that users haven't classified yet.  
##Models:  
A model defines the relationship between features and label.  
**Example:** the spam detector may associate certain features strongly with "spam"  
####Training:  
Training means creating or learning the model. The model is shown labeled examples which enables the model to gradually
learn the relationship between the features and label.  
####Inference:  
Inference is applying the trained model to unlabeled examples. The trained model is used to make predictions, which can
be denoted as **y<sup>'</sup>**  
####Regression Models:  
A regression model predicts **continuous** values.  
**Example:** what is the value of a house?, what is the probability the user will take x action?
####Classification Models:  
A classification model predicts **discrete** values.  
**Example:** is a given email message spam or not spam?, is this an image of a dog, a cat, or a hamster?
##Linear Regression:  
A linear relationship can be defined with the formula
* y = mx + c

_y = value being predicted, m = slope of the line, x = input feature, c = y-intercept_  
  
This formula is written differently in machine learning as:
* y<sup>'</sup> = b + w<sub>1</sub>x<sub>1</sub>

_y<sup>'</sup> = the predicted label (desired output), b = bias (y-intercept), w<sub>1</sub> = weight of feature 1 (same
concept as slope of a line m), x<sub>1</sub> = feature (a known input)_  
  
More sophisticated models will have more than one feature, each feature will have its own weight, for example a model
that has three features will have the following formula:  
* y<sup>'</sup> = b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + w<sub>3</sub>x<sub>3</sub>  

####Training: 
Training a model means learning good values for all the weights and bias from labeled examples. In supervised learning 
the ML algorithm builds a model bu examining many examples and attempting to find a model that minimizes loss, this
process is called **empirical risk minimization**.
####Loss: 
Loss is the penalty for a bad prediction. Loss is a number indicating how bad the models prediction was on a single
example. A perfect model has 0 loss as there would be no difference between predicted and actual values. The goal of 
training a model is the find a set of weights for each feature that have low loss across all examples. This can be
visualised as how far data points are away from a line of best fit on a graph, the distance from the point to the line
is the loss. loss is always a positive value.
####Squared Loss: 
Linear regression models commonly use squared loss (also known as L<sub>2</sub> loss). The squared loss for a single
example is:  
* the square of the difference between the label and the prediction
* (observation - prediction(x))<sup>2</sup>
* (y - y<sup>'</sup>)<sup>2</sup>
####Mean Square Error (MSE):  
MSE is the average squared loss per example over the whole data set. To calculate the MSE, sum all the squared losses
for individual examples and then divide by the number of examples:    

<img src="https://latex.codecogs.com/gif.latex?MSE&space;=&space;\frac{1}{N}\sum_{(x,y)\in&space;D}^{}(y&space;-&space;prediction(x))^2" title="MSE = \frac{1}{N}\sum_{(x,y)\in D}^{}(y - prediction(x))^2" />
 
 _where (x,y) is an example where x = set of features, y = examples labels. prediction(x) = function of weights and bias
 in combination with the set of features x, D = data set containing many labeled examples, which are (x,y) pairs, N = 
 number of examples in D_  
 