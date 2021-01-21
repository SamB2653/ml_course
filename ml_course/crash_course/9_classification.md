#Classification:  
####Thresholding:
Logistic regression returns a probability. You can use the returned probability "as is" (for example, the probability
that the user will click on this ad is 0.00023) or convert the returned probability to a binary value (for example,
this email is spam).  

A logistic regression model that returns 0.9995 for a particular email message is predicting that it is very likely
to be spam. Conversely, another email message with a prediction score of 0.0003 on that same logistic regression 
model is very likely not spam.

However, what about an email message with a prediction score of 0.6? In order to map a logistic regression value to a
binary category, you must define a classification threshold (also called the decision threshold). A value above that
threshold indicates "spam"; a value below indicates "not spam." It is tempting to assume that the classification 
threshold should always be 0.5, but thresholds are problem-dependent, and are therefore values that you must tune.  

Tuning a threshold for logistic regression is different from tuning hyperparameters such as learning rate. It determines
how bad a mistake will be, for example it is far worse to mark non spam messages as spam than the other way around.

####True vs. False and Positive vs. Negative:  
There are 4 possible outcomes of a prediction (can be shown in a 2x2 confusion matrix):
* **True Positive (TP):** the model correctly predicts the positive class
* **False Positive (FP):** the model incorrectly predicts the positive class
* **True Negative (TN):** the model correctly predicts the negative class.
* **False Negative (FN):** the model incorrectly predicts the negative class.

####Accuracy:  
Accuracy is the fraction of predictions the model got right:
* Accuracy = Number of correct predictions / Total number of predictions

Accuracy for binary classification:
* Accuracy = ( TP + TN ) / ( TP + TN +FP + FN ) 

Accuracy alone doesn't tell the full story when you're working with a class-imbalanced data set, like one where 
there is a significant disparity between the number of positive and negative labels.

####Precision and Recall:  
Precision is defined as:
* Precision = TP / TP + FP

Recall is defined as:
* Recall = TP / TP + FN

To fully evaluate the effectiveness of a model, you must examine both precision and recall. Usually improving one will 
decrease the other.  
**Example:** consider spam email classification where TP = 8, TN = 17, FP = 2 and FN = 3.  

Precision measures the percentage of emails flagged as spam that were correctly classified:
* Precision = 8 / ( 8 + 2 ) = 0.8  

Recall measures the percentage of actual spam emails that were correctly classified:
* Recall = 8 / ( 8 + 3 ) = 0.74

If the classification is increased this will change the precision and recall scores. so:  
TP = 7, TN = 18, FP = 1 and FN = 4.  

Precision increases:
* Precision = 7 / ( 7 + 1 ) = 0.88  

Recall decreases:
* Recall = 7 / ( 7 + 4 ) = 0.64

The effect will be the opposite if the classification threshold is lowered. F1 score is a good way of combining the 
precision and recall of a model into one value (harmonic mean of precision and recall).

####ROC curve: 
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at 
all classification thresholds. This curve plots two parameters:
* True Positive Rate
* False Positive Rate

![alt text](https://developers.google.com/machine-learning/crash-course/images/ROCCurve.svg
"TP vs. FP rate at different classification thresholds")

####AUC: Area Under the ROC Curve:
AUC measures the entire two-dimensional area underneath the entire ROC curve. AUC provides an aggregate measure of
performance across all possible classification thresholds. One way of interpreting AUC is as the probability that the 
model ranks a random positive example more highly than a random negative example. A model whose predictions are 100% 
wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.  

AUC is desirable for the following two reasons:
* AUC is **scale-invariant**. It measures how well predictions are ranked, rather than their absolute values.
* AUC is **classification-threshold-invariant**. It measures the quality of the model's predictions irrespective of what 
classification threshold is chosen.

There are limits of AUC:
* **Scale invariance is not always desirable**. For example, sometimes we really do need well calibrated probability
outputs, and AUC won’t tell us about that.
* **Classification-threshold invariance is not always desirable**. In cases where there are wide disparities in the 
cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For
example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that 
results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.

####Prediction Bias:
Logistic regression predictions should be unbiased, so the average of predictions should equal the average of
observations. Prediction bias is the measure of how far apart these averages are:
* _prediction bias = average of predictions - average of labels in data set

Possible causes of prediction bias are:
* Incomplete feature set
* Noisy data set
* Buggy pipeline
* Biased training sample
* Overly strong regularization

Adding a calibration layer is a bad idea, for example if there is a +3% bias adding a layer that reduces it by 3%. This
is because:
* You're fixing the symptom rather than the cause.
* You've built a more brittle system that you must now keep up to date.

Calibration layers should be avoided and a good model will have a near zero prediction bias. However prediction bias
does not indicate on its own that the model is good as for example just predicting the mean will give a prediction 
bias of 0.

####Prediction Bias - Bucketing:
Logistic regression predicts a value between 0 and 1. However, all labeled examples are either exactly 0 (meaning, 
for example, "not spam") or exactly 1 (meaning, for example, "spam"). Therefore, when examining prediction bias, you 
cannot accurately determine the prediction bias based on only one example; you must examine the prediction bias on a 
"bucket" of examples. prediction bias for logistic regression only makes sense when grouping enough examples together 
to be able to compare a predicted value (for example, 0.392) to observed values (for example, 0.394).  

Buckets can be formed by:
* Linearly breaking up the target predictions.
* Forming quantiles.

####binary_classification.py:
Binary classification in TensorFlow example.
* Convert a regression question into a classification question.
* Modify the classification threshold and determine how that modification influences the model.
* Experiment with different classification metrics to determine your model's effectiveness.
