# Bucketing: 
If you choose to bucket your numerical features, be clear about how you are setting the boundaries and which type of
bucketing you’re applying:

* **Buckets with equally spaced boundaries:** The boundaries are fixed and encompass the same range (for example, 
0-4 degrees, 5-9 degrees, and 10-14 degrees, or $5,000-$9,999, $10,000-$14,999, and $15,000-$19,999). Some buckets 
could contain many points, while others could have few or none.
* **Buckets with quantile boundaries:** Each bucket has the same number of points. The boundaries are not fixed and 
could encompass a narrow or wide span of values.



#### Buckets (Equally Spaced):
The data can be divided into buckets of equal width, then each individual bucket can tell something about the data set. 
The numerical values are converted to categorical values of equal spacing.

![alt text](https://developers.google.com/machine-learning/data-prep/images/bucketizing-needed.svg?dcb_=0.4238748845289071
"Buckets (Equally Spaced)")  

#### Quantile Bucketing:
With some distributions equally spaced buckets don’t capture the distribution very well. The buckets are created with 
the same number of points instead of equal intervals within the data set. Some will be a very small span but others will 
have a very large one.

![alt text](https://developers.google.com/machine-learning/data-prep/images/bucketizing-applied.svg?dcb_=0.23617561919696128
"Quantile Bucketing")  

Bucketing with equally spaced boundaries is an easy method that works for a lot of data distributions. For skewed data, 
however, quantile bucketing is the better option.
