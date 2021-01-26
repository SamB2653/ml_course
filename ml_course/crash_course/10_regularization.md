# Regularization:  
#### L<sub>2</sub> Regularization:
The graph below shows a generalization curve, which shows the loss for both the training set and validation set against 
the number of training iterations.

![alt text](https://developers.google.com/machine-learning/crash-course/images/RegularizationTwoLossFunctions.svg
"Generalization curve")

Loss is always decreasing but at some point the validation loss eventually starts to increase. This means the model
is over fitting as the model is fitting to the nuances of the training data set and not for the general data set.
Over fitting can be prevented by penalizing complex models, a principle called regularization.  

**Empirical Risk Minimization (minimize loss):**
* _minimize(Loss(Data|Model))_

**Structural Risk Minimization (minimize loss + complexity):**
* _minimize(Loss(Data|Model) + complexity(Model))_

The training optimization algorithm is now a function of two terms;  the loss term, which measures how well the model 
fits the data, and the regularization term, which measures model complexity. Model complexity is commonly measured in
two ways:
* as a function of the weights of all the features in the model.
* as a function of the total number of features with nonzero weights.

If model complexity is a function of weights, a feature weight with a high absolute value is more complex than a 
feature weight with a low absolute value. Complexity can be quantified using the L<sub>2</sub> regularization formula,
which defines the regularization term as the sum of all the feature weights:
* _L<sub>2</sub> regularization term = ||w||<sub>2</sub><sup>2</sup> = w<sub>1</sub><sup>2</sup> + w<sub>2</sub>
<sup>2</sup> + ... + w<sub>n</sub><sup>2</sup>_

In the formula weights close to zero have little effect on model complexity, while outlier weights can have a huge 
impact.  
**Example:** a linear model with the following weights:  
* { w<sub>1</sub> = 0.1, w<sub>2</sub> = **5**, w<sub>3</sub> = 0.5, w<sub>4</sub> = 0.2 }

* { w<sub>1</sub><sup>2</sup> = 0.01, w<sub>2</sub><sup>2</sup> = **25**, w<sub>3</sub><sup>2</sup> = 0.25, w<sub>
4</sub><sup>2</sup> = 0.03 }

* L<sub>2</sub> = 25.3

So w<sub>2</sub><sup>2</sup> contributes 25, so is nearly all of the complexity. The sum of the other values is only
0.3.

#### Regularization Rate Lambda (λ):
You can tune the overall impact of the regularization term by multiplying its value by a scalar known as lambda 
(also called the regularization rate). Increasing the lambda value strengthens the regularization effect. 
This means the previous formula becomes:

* _minimize(Loss(Data|Model) + **λ**complexity(Model))_

Performing L<sub>2</sub> regularization has the following effect on a model:
* Encourages weight values toward 0 (but not exactly 0).
* Encourages the mean of the weights toward 0, with a normal (bell-shaped or Gaussian) distribution.

When choosing a lambda value, the goal is to strike the right balance between simplicity and training-data fit:
* If your lambda value is too high, your model will be simple, but you run the risk of underfitting your data. Your
 model won't learn enough about the training data to make useful predictions.
* If your lambda value is too low, your model will be more complex, and you run the risk of overfitting your data. 
Your model will learn too much about the particularities of the training data, and won't be able to generalize to
new data.

The ideal value of lambda produces a model that generalizes well to new, previously unseen data. However this is data
dependant so needs tuning.

#### L<sub>2</sub> Regularization and Learning Rate:
There's a close connection between learning rate and lambda. Strong L<sub>2</sub> regularization values tend to drive
feature weights towards 0, lowering learning rates with early stopping often produces the same effect as the steps away
from 0 are not as large. Consequently you should not change learning rate and lambda at the same time or the results
may be confusing.

#### Early Stopping:
Early stopping is a method for regularization that involves ending model training before training loss finishes 
decreasing. In early stopping, you end model training when the loss on a validation dataset starts to increase, that
is, when generalization performance worsens.   

In reality there will be some form of early stopping when training in a continuous fashion. Some trends in the data 
wont have had enough time to form links in the model. In practice (when training across a fixed batch of data) give 
yourself a high enough number of iterations that early stopping doesn't play into things.
