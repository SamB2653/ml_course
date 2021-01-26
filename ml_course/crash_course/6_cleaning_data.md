# Cleaning Data:  
Bad and damaged data must be removed or repaired before a model is trained, bad data can skew the model.
#### Scaling Feature Values:
Scaling means converting floating-point feature values from their natural range (for example, **100 to 900**) into a 
standard range (for example, **0 to 1** or **-1 to +1**). If a feature set consists of only a single feature, then 
scaling provides little to no practical benefit. If, however, a feature set consists of multiple features, 
then feature scaling provides the following benefits:
* Helps gradient descent converge more quickly.
* Helps avoid the "NaN trap," in which one number in the model becomes a NaN (e.g., when a value exceeds the 
floating-point precision limit during training), and—due to math operations—every other number in the model also
eventually becomes a NaN.
* Helps the model learn appropriate weights for each feature. Without feature scaling, the model will pay too much 
attention to the features having a wider range.

You don't have to give every floating-point feature exactly the same scale. Nothing terrible will happen if Feature A
is scaled from -1 to +1 while Feature B is scaled from -3 to +3. However, your model will react poorly if Feature B
is scaled from 5000 to 100000.

#### Scaling Methods (Z-Score):
One simple way to scale numerical data is to linearly map [min value, max value] to a small scale, such as [-1, +1].
Another method is to calculate the Z score of each value. The Z score relates the number of standard deviations
away from the mean, so:
* _scaledvalue = (value - mean) / stddev_

**Example:** if mean = 100, standard deviation = 20 and original value = 130 then:
* scaledvalue = (130 - 100) / 20
* scaledvalue = 1.5

Scaling with Z scores means that most scaled values will be between -3 and 3, but a few values will be a little higher
or lower than that range.

#### Handling Extreme Outliers:
When collecting data most data should be within a few standard deviations of the mean, however there could be some 
extreme outliers that skew the data. There are multiple ways to deal with this:
* **Take the log of every value (Log scaling):**  
Log scaling will reduce the distance the outlier is from the mean, however the outlier will still be present in the
data set and will be a significant distance from the other, desired data.
* **Modify values greater than a set range (Clipping Feature Values):**  
Clipping the data at a set limit means that any value over the selected value will be set equal to the upper bound.
This means that there will be a small peak at the limit where all the outliers have been set to this new value, this 
isn't ideal but is better than including just the raw data into the model.
* **Put ranges of numerical data into categories (Binning / Bucketing):**  
Divide the numerical data into set "bins", each of the same range so create distinct boolean features. These specific 
bins can be combined into a single feature vector (via one-hot encoding). Therefore each bin will have a specific
weighting in the model. You can also bin by quantile, this removes the need to worry about outliers.

#### Scrubbing:
The following examples can be reasons the data is untrustworthy:
* **Omitted values**. For instance, a person forgot to enter a value for a house's age.
* **Duplicate examples**. For example, a server mistakenly uploaded the same logs twice.
* **Bad labels**. For instance, a person mislabeled a picture of an oak tree as a maple.
* **Bad feature values**. For example, someone typed in an extra digit, or a thermometer was left out in the sun.

Bad examples can be fixed by removing them from the data set. It is simple to detect omitted or duplicated values but
detecting bad labels and feature values is much more tricky. A good way to check is via graphing, most commonly via a
histogram. The following information can help determine bad data:
* Maximum and minimum
* Mean and median
* Standard deviation

You can look at these and by eye see if the results are expected. The key rules are:
* Keep in mind what you think your data should look like.
* Verify that the data meets these expectations (or that you can explain why it doesnt).
* Double-check that the training data agrees with other sources (for example, dashboards).
