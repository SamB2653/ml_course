# Normalisation: 
The goal of normalization is to transform features to be on a similar scale. This improves the performance and training 
stability of the model.

#### Normalization Techniques:
Four common normalization techniques may be useful:  
* Scaling to a range: When the feature is more-or-less uniformly distributed across a fixed range
* Feature Clipping: When the feature contains some extreme outliers
* Log scaling: When the feature conforms to the power law
* Z-score: When the feature distribution does not contain extreme outliers

![alt text](https://developers.google.com/machine-learning/data-prep/images/normalizations-at-a-glance-v2.svg
"normalisation Techniques")  

#### Scaling to a Range:  
scaling means converting floating-point feature values from their natural range (for example, 100 to 1600) into a 
standard range, usually 0 and 1 (or sometimes -1 to +1). Use the following simple formula to scale to a range:

<img src="https://latex.codecogs.com/png.latex?x%5E%7B%27%7D%3D%5Cfrac%7B%28x-x_%7Bmin%7D%29%7D%7B%28x_%7Bmax%7D-x_%7Bmin%7D%29%7D" title="Equation" />
  
This is a good choice when the following 2 conditions are met:  
* Approximate upper and lower bounds on the data are known with few or no outliers
* The data is approximately uniformly distributed across that range  

A good use would be ages as most fall between 0 and 90, every part of the range has many values. A bad use would be for 
income as only a few people have a very high income and most people would be contained in a small part of the data.

#### https://developers.google.com/machine-learning/data-prep/images/norm-clipping-outliers.svg:  
If the data set contains extreme outliers then feature clipping can be used, it caps feature values above or below set 
thresholds. So for examples all values over a set threshold of 50 would be set to exactly 50. This can be done before 
or after other normalisations.

![alt text](https://developers.google.com/machine-learning/data-prep/images/norm-clipping-outliers.svg
"Feature Clipping")  

Another clipping strategy can be to clip by z-score to a set number of standard deviations, eg 3 standard deviations.

#### Log Scaling:  
Log scaling computes the log of the values to compress a wide range to a narrow range:

<img src="https://latex.codecogs.com/png.latex?x%5E%7B%27%7D%3Dlog%28x%29" title="Equation" />
  
Log scaling is helpful when a handful of the values have many points, while most other values have few points. This is 
the power law distribution. A good example is movie ratings as shown in the example below, most movies have very few 
ratings (data in the tail), while a few have lots of ratings (data in the head). Log scaling changes the distribution, 
helping to improve linear model performance.

![alt text](https://developers.google.com/machine-learning/data-prep/images/norm-log-scaling-movie-ratings.svg
"Log Scaling")  

#### Z-Score:  
Z-score is a variation of scaling that represents the number of standard deviations away from the mean. Use z-score to 
ensure feature distributions have mean = 0 and std = 1. It’s useful when there are a few outliers, but not so extreme 
that clipping is needed:  

<img src="https://latex.codecogs.com/png.latex?x%5E%7B%27%7D%3D%5Cfrac%7B%28x-%5Cmu%29%7D%7B%5Csigma%7D" title="Equation" />

Note: μ = mean and σ = standard deviation  

In the example z-score squeezes raw values that have a range of ~40000 down into a range from roughly -1 to +4. If you 
are unsure if the outliers are truly extreme start with z-score unless you have feature values that you don't want the 
model to learn; for example, the values are the result of measurement error or a quirk.
