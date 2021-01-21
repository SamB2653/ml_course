#TensorFlow:  
####Introduction:
TensorFlow is an end-to-end open source platform for machine learning. TensorFlow is a rich system for managing all 
aspects of a machine learning system. TensorFlow APIs are arranged hierarchically, with the high-level APIs built on 
the low-level APIs. Machine learning researchers use the low-level APIs to create and explore new machine learning 
algorithms.

![alt text](https://developers.google.com/machine-learning/crash-course/images/TFHierarchyNew.svg 
"Hierarchy of TensorFlow Toolkits")

####Hyperparameter Tuning:  
* Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches 
or approaches zero.
* If the training loss does not converge, train for more epochs.
* If the training loss decreases too slowly, increase the learning rate. Note that setting the learning rate too high 
may also prevent training loss from converging.
* If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate
* Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
* Setting the batch size to a very small batch number can also cause instability. First, try large batch size values. 
Then, decrease the batch size until you see degradation.
* For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory.
 In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.
* Hyperparameters are data dependant, so experimentation is needed.

####Overfitting (Generalization):  
A model with a low loss can still be a bad model, this is due to over fitting test data. If new data is added to an
over fitted model then the model will adapt poorly to the new data. This is usually caused by creating a model that
is too complex, models should aim to fir the data as simply as possible to avoid over fitting. Machine learning's goal
is to predict well on new data drawn from a (hidden) true probability distribution, the model only sees a sample from
the training data set:
>"The less complex an ML model, the more likely that a good empirical result is not just due to the
 peculiarities of the sample." - Ockham's razor 

Generalization bounds are a statistical description of a model's ability to generalize to new data based on 
factors such as:
* The complexity of the model
* The model's performance on training data

The best way to avoid over fitting is to split the data into train and test sets. The model will use the training data
set taken from all the data to fit the model while the model is not trained on the test subset of data. The testing 
data can be used to determine how the model reacts to unseen data.

The basic assumptions that guide generalization are:
*  We draw examples independently and identically (i.i.d) at random from the distribution. In other words, examples 
don't influence each other. (An alternate explanation: i.i.d. is a way of referring to the randomness of variables.)
* The distribution is stationary; that is the distribution doesn't change within the data set.
* We draw examples from partitions from the same distribution.

These can be violated and when we know that any of the preceding three basic assumptions are violated, we
must pay careful attention to metrics. Examples of these assumptions being violated:
* Consider a model that chooses ads to display. The i.i.d. assumption would be violated if the model bases its choice 
of ads, in part, on what ads the user has previously seen.
* Consider a data set that contains retail sales information for a year. User's purchases change seasonally,
 which would violate stationarity.

####linear_regression_synthetic.py:
Explores linear regression with a toy dataset.  
Tune the following hyperparameters:
* Learning rate
* Number of epochs
* Batch size
* Interpret different kinds of loss curves.

####linear_regression_real.py:
Looks at the kind of analysis you should do on a real dataset.
Looks at the following:
* Read a .csv file into a pandas DataFrame.
* Examine a dataset.
* Experiment with different features in building a model.
* Tune the model's hyperparameters.
