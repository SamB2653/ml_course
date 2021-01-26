# Partitioning Data Sets:  
#### Train Test Sets:
A data set should be split into train and test data subsets. The training data subset is used for training the model 
while the testing subset is used for testing the trained model and cannot be seen by the model during training. There
are two main conditions for the test set:
* Is large enough to yield statistically meaningful results.
* Is representative of the data set as a whole. In other words, don't pick a test set with different characteristics 
than the training set.

The test set serves as a proxy for new data. If a model does about as well on the test data as it does on the 
training data then there isn't over fitting on the training data. You should never train on your testing data, the 
training data should be checked to see if there are any duplicate values to remove any accidental bias.

#### Validation Sets:
Instead of just using a train test data split you can have another additional subset of data called validation data. 
With train test the model is trained with the training data, then evaluated with the test data, tweak the model 
from the test set and pick the model that does best on the test set. 

Now with validation data the new workflow will be training on the training data,
evaluating the model with the validation data, tweak the model from the validation set, pick the model that works the
best on the validation set and then confirm the results on the testing set.

Using a validation set greatly reduces the chance of over fitting as the test set is exposed much less to the model.
In an ideal model training, validation and test RMSE should be similar.

#### validation_and_test.py:
* Split a training set into a smaller training set and a validation set.
* Analyze deltas between training set and validation set results.
* Test the trained model with a test set to determine whether your trained model is over fitting.
* Detect and fix common training problems.