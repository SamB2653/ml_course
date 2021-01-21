#Data Engineering:  
####Mapping Raw Data to Features:
In traditional programming, the focus is on code. In machine learning projects, the focus shifts to representation. 
A model can be improved greatly by adding and improving its features.

Raw data should be converted to feature vectors, this is a set of floating-point values derived from the raw data set.
Many machine learning models must represent the features as real-numbered vectors since the feature values must be 
multiplied by the model weights.

####Mapping Numeric Values:
Integer and floating-point data does not need a special encoding because they can be multiplied by a numeric weight.  
**Example:** the raw integer **5** can be easily converted to a floating point value of **5.0**.
* num_rooms: **5** ----> num_rooms_feature = **[ 5.0 ]**

####Mapping Categorical  Values - Bucketing (Binning):
Categorical features have a discrete set of possible values. Models weights can't be multiplied by strings so the string
values must be converted to numeric values. This can be accomplished by bucketing or one-hot encoding.

Bucketing is where the values are split into distinct categories by defining a mapping from the feature values.
**Example:** the data set {'Duck Road', 'Dog Boulevard', 'Shorebird Way', 'Badger Avenue'} can be mapped to 
numerical values with an addition of an other category to catch any other values.
* {'Duck Road', 'Dog Boulevard', 'Shorebird Way', 'Badger Avenue'} ---->
* {'Duck Road': **0**, 'Dog Boulevard': **1**, 'Shorebird Way': **2**, 'Badger Avenue': **3**, 'Other': 
**4**}

There are issues with bucketing, such as:
* The model learns a single weight that applies to all streets, so if the street_name weight is **6** then 
Duck Road will be (6 * 0), Dog Boulevard will be (6 * 1) ect... It is assumed that the street names have been
ordered by their price in a linear pattern which is unlikely. The model needs the flexibility of learning different 
weights for each street that will be added to the price estimated using the other features.
* The model doesnt accounting for cases where street_name may take multiple values, for example if the house is on a 
corner of two streets. There could be no way of encoding this if street_name contains a single index.

####Mapping Categorical  Values - One-Hot Encoding:
To remove both the constraints of bucketing, we can instead create a binary vector for each categorical feature in 
our model that represents values as follows:
* For values that apply to the example, set corresponding vector elements to 1.
* Set all other elements to 0.

The length of this vector is equal to the number of elements in the vocabulary. If there is just a single value of 1
then it is called one-hot encoding, if there are multiple values of 1 then it is called multi-hot encoding.  
**Example:** Shorebird Way can be converted to to a binary vector where Shorebird Way has a value of 1 and all other
streets have a value of 0.
* street_name: **"Shorebird Way"** ----> street_name_feature = **[0, 0, 1, 0, 0]**

This effectively creates a boolean value foe every feature, therefore the model can use the weight for each individual
street name. If there is a house on a corner then two binary values can be set to 1, the model can then use both
their respective weights.

####Sparse and Dense Representations:
Suppose that you had 1,000,000 different street names in your data set that you wanted to include as values for 
street_name. Explicitly creating a binary vector of 1,000,000 elements where only 1 or 2 elements are true is a
very inefficient representation in terms of both storage and computation time when processing these vectors.
In this situation, a common approach is to use a sparse representation in which only nonzero values are stored. 
In sparse representations, an independent model weight is still learned for each feature value.

A dense representation of 1,000,000 different street street names would also store the 0 integers in comparison to 
sparse representation where only the nonzero values are stored.

Data set (multi-hot encoded):
* [ "dog": 0, "cat": 0, "chicken": 1, "cow": 0 , "rat": 0, "squirrel": 1 ]

Dense representation:
* [ "dog": 0, "cat": 0, "chicken": 1, "cow": 0 , "rat": 0, "squirrel": 1 ]

Sparse representation:
* [ "chicken": 1, "squirrel": 1 ]

####Qualities of Good Features:
* **Avoid rarely used discrete feature values**:  
A good feature should appear more than just a few times in the data set.
This enables the model to learn how this feature relates to the label, many examples of a feature with similar values
allows the model to determine what is a good predictor for the label.
     
    An example of a good feature would be **house_type: victorian** when looking at house types rather than an identifier
such as **unique_house_id: 91hgd937h1d**. The identifier unique_house_id would be a bad feature as the model cant
learn anything from a unique value.  

* **Prefer clear and obvious meanings**:  
Each feature should have a clear and obvious meaning to anyone on the project. 

    For example it makes sense when looking
at **house_age_years: 27** instead of **house_age_seconds: 852055200**. Also this will help when identifying noisy 
data as if the house age is something like 2700 then it is clearly incorrect, however it would be more unclear if this
was not in straight forward units.

* **Don't mix "magic" values with actual data**:  
Good floating-point features don't contain peculiar out-of-range discontinuities or "magic" values. 

    For example, suppose a feature holds a floating-point value between **0 and 1**. Good values would be 
    **quality_rating: 0.45** or **quality_rating: 0.76** ect... A bad value would be a value of **quality_rating: -1**
    To explicitly mark magic values, create a Boolean feature that indicates whether or not a **quality_rating** was 
    supplied. Give this Boolean feature a name like **is_quality_rating_defined**, then for the quality_rating
    feature replace the magic values as follows:
    
    * For variables that take a finite set of values (discrete variables), add a new value to the set and use it to 
    signify that the feature value is missing.
    
    * For continuous variables, ensure missing values do not affect the model by using the mean value of the 
    feature's data.

* **Account for upstream instability**:  
  The definition of a feature should not change over time, for example a city name is unlikely to change. However if
  the data comes from an external source and is provided in a format such as **city_zone_id: 8927** then that source 
  naming convention could change.