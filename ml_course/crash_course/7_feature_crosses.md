# Feature Crosses:  
Linear models can be used to solve non linear problems by crossing features together. A feature cross is a synthetic 
feature that encodes non-linearity in the feature space by multiplying two or more input features together (cross
product).  
**Example:** you can create a new feature **x<sub>3</sub>** from feature **x<sub>1</sub>** and **x<sub>2</sub>**
* x<sub>3</sub> = x<sub>1</sub> x<sub>2</sub>

The linear formula would now include x<sub>3</sub>:
* y = b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + w<sub>3</sub>x<sub>3</sub> 

The linear model will learn the weight of w<sub>3</sub> and will encode non-linear information, the linear model doesnt
need any changes to learn this.

#### Types of Feature Crosses:
We can create many different kinds of feature crosses. For example:
* **[A X B]**: a feature cross formed by multiplying the values of two features.
* **[A x B x C x D x E]**: a feature cross formed by multiplying the values of five features.
* **[A x A]**: a feature cross formed by squaring a single feature.

Using stochastic gradient descent, linear models can be trained efficiently. Consequently, supplementing scaled 
linear models with feature crosses has traditionally been an efficient way to train on massive-scale data sets.

#### Crossing One-Hot Vectors:
Machine learning models frequently cross one-hot feature vectors.  
**Example:** one-hot encoding **country** and **language** features generates vectors with binary features that can be
interpreted as:  
* country = USA, country = France OR language = English, language = Spanish

Conducting a feature cross of these one-hot encodings, you get binary features that can be interpreted as logical
conjunctions:

* country: USA AND language: Spanish

**Example:** if you bin latitude and longitude to produce separate one-hot five-element feature vectors, longitude and
latitude could be represented as:

* binned_latitude = [0, 0, 0, 1, 0]
* binned_longitude = [0, 1, 0, 0, 0]

Conduct a feature cross of these two feature vectors:

* binned_latitude X binned_longitude

This will create a 25 element one-hot vector (24 zeroes and 1 one). The single 1 value identifies a particular
conjunction of latitude and longitude, associations can then be learned from these conjunctions.

**Example:** A more complex latitude longitude example:

<pre><code>binned_latitude(lat) = [
  0  < lat <= 10
  10 < lat <= 20
  20 < lat <= 30
]

binned_longitude(lon) = [
  0  < lon <= 15
  15 < lon <= 30
]</code></pre>

Conducting a feature cross of these bins means they will become:

<pre><code>binned_latitude_X_longitude(lat, lon) = [
  0  < lat <= 10 AND 0  < lon <= 15
  0  < lat <= 10 AND 15 < lon <= 30
  10 < lat <= 20 AND 0  < lon <= 15
  10 < lat <= 20 AND 15 < lon <= 30
  20 < lat <= 30 AND 0  < lon <= 15
  20 < lat <= 30 AND 15 < lon <= 30
]</code></pre>

#### Overcrossing:
Adding too many feature crosses can cause the model to over fit the training data and be too complex. It is best to
keep the model as simple as possible to avoid these issues.

#### feature_cross.py:
Look at how to do a feature cross on data.
* Use tf.feature_column methods to represent features in different ways.
* Represent features as bins.
* Cross bins to create a feature cross.