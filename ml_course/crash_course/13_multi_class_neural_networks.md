#Multi-Class Neural Networks:  
Binary classification models can pick between one of two possible choices. Multi-class classification can pick from 
multiple possible choices.

####Binary vs Multi-class classification:  
Binary classification models determine if the outcome is one of two possible choice, is or isn't. For example if an email
is or is not spam ect. Multi-class classification models determine between multiple possibilities, such as what type
of car or species of flower ect. In practice multi-class models can have millions of separate classes.

####One vs All:  
A one vs all solution can solve a classification problem with N solutions, the solution consists of N separate binary
classifiers, one for each possible outcome. When training, the model runs through a sequence of binary classifiers and 
trains each to answer a separate classification question. **Example**: If you were training a model to recognise a 
human then you could train 5 different recognizers, 4 negative that are not a human and 1 positive that is a human:
* Is the image a Dog? **NO**  
* Is the image a Cat? **NO**  
* Is the image a Helicopter? **NO**  
* Is the image a Human? **YES**  
* Is the image a Cheese? **NO**  
This is a good approach when the number of classes is small, however is inefficient when the number of classes increases.
The solution to this is to create a deep neural network where each output node represents a different class

![alt text](https://developers.google.com/machine-learning/crash-course/images/OneVsAll.svg "one vs all")

####Softmax:  
Logistic regression would produce a decimal between 0 and 1, if the output was 0.7, there would be a 70% chance of a
positive result and a 30% chance of a negative result, the sum of probabilities could not exceed 1. Softmax extends 
this to multi class problems as decimal probabilities are assigned to each class within the multi-class problem. These
decimal probabilities must add up to 1. This constraint allows the training to converge more quickly.**Example**: Softmax
could produce the following probabilities:
* Is the image a Dog? **0.04**  
* Is the image a Cat? **0.02**  
* Is the image a Helicopter? **0.02**  
* Is the image a Human? **0.90**  
* Is the image a Cheese? **0.02**
Softmax is implemented in a layer just before the output, it must have the same number of nodes as the output layer.  

The following Softmax formula extends the formula for logistic regression into multiple classes:

<img src="https://latex.codecogs.com/gif.latex?p(y=j|x)=\frac{e^{(w_{j}^{T}x&plus;b_{j})}}{\sum_{k\in&space;K^{(w_{k}^{T}x&plus;b_{k})}}}" title="Softmax Equation" />
 
####Softmax Options:  
* **Full Softmax** calculates a probability for every possible class. 
* **Candidate sampling** calculates a probability for all the positive labels but only for a random sample of negative
labels. For example, if we are determining whether an input image is a human or dog, we don't have to provide 
probabilities for every non-human example.

Full Softmax is viable when there are low number of classes but becomes much more expensive as the number of classes
increases. Candidate sampling is the solution to this issue as it improves efficiency with a large number of classes.

####One Label vs Many Labels:
Softmax assumes that each example is a member of exactly one class. Some examples can simultaneously be a member of 
multiple classes. For many label examples you cannot use Softmax and must rely on logistic regression. For example if
you are detecting apples within an image, Softmax will work if images are of a singular apple, however if the image
contains apples and bananas then multiple logistic regressions will have to be used instead.

####softmax.py:
Use Softmax with Tensorflow to develop a model that classifies hand written digits:
* Create a deep neural network that performs multi-class classification.
* Tune the deep neural network.
* Understand the classic MNIST problem:
    * The MNIST training set contains 60,000 examples.
    * The MNIST test set contains 10,000 examples.
    * There are 10 output classes (1 for each digit)
    * A pixel map is used to determine how a person wrote a digit:
    
![alt text](https://www.tensorflow.org/images/MNIST-Matrix.png "weight vs loss graph")
    