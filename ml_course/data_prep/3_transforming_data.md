# Transforming Data: 
Feature engineering is the process of determining which features might be useful in training a model, and then creating 
those features by transforming raw data found in log files and other sources. Numeric and categorical data can be 
transformed, there are trade offs to different approaches.

#### Reasons for Data Transformation:
There are various reasons for transforming features of data:
* **Mandatory transformations** for data compatibility:
    * Converting non-numeric features into numeric. Matrix multiplication cannot be done on a string, so must be 
    converted to some numeric representation.
    * Resizing inputs to a fixed size. Linear models and feed-forward neural networks have a fixed number of input nodes, 
    so the input data must always have the same size.
* **Optional quality transformations** that may help the model perform better:
    * Tokenization or lower-casing of text features
    * Normalized numeric features (most models perform better afterwards)
    * Allowing linear models to introduce non-linearities into the feature space

#### Where to Transform:
Transformations can be applied either while generating the data on disk, or within the model. There are pros and cons 
of each method:
* **Transforming prior to training:** Transformation before training. This code lives separate from the machine 
learning model.
    * Computation is performed only once
    * Computation can look at the entire data set to determine the transformation
    * Transformations need to be reproduced at prediction time, this can cause skew
    * Any transformation changes require rerunning the data generation, leading to slower iterations
    
    In offline serving, the code that generates the training data could be reused. In online serving, the code that 
    creates the data set and the code used to handle live traffic are almost necessarily different, which makes it 
    easy to introduce skew.
    
* **Transforming within the model:**  The transformation is part of the model code. The model takes in untransformed 
data as input and will transform it within the model.
    * Easy iterations. If you change the transformations, you can still use the same data files
    * You're guaranteed the same transformations at training and prediction time
    * Expensive transforms can increase model latency
    * Transformations are per batch  
    
    There are many considerations for transforming per batch. If you want to normalize a feature by its average 
    value-that is, you want to change the feature values to have mean 0 and standard deviation 1. When transforming 
    inside the model, this normalization will have access to only one batch of data, not the full data set. You can 
    either normalize by the average value within a batch (dangerous if batches are highly variant), or pre-compute 
    the average and fix it as a constant in the model.  
    
Before transformation can take place the data should be explored and cleaned. The following tasks should be completed:  
* Examine several rows of data
* Check basic statistics
* Fix missing numerical entries
* Visualise the data via scatter or histogram plots, view these throughout the pipeline to help see major changes

#### Transforming Numeric Data:  
There are two kinds of common transformations for numeric data:
* **Normalizing:** Transforming numeric data to the same scale as other numeric data
* **Bucketing:** Transforming numeric (usually continuous) data to categorical data  

Normalization is necessary if there are very different values within the same feature (for example, city population). 
Without normalization, the training could blow up with NaNs if the gradient update is too large. There could be two 
different features with widely different ranges (e.g., age and income), causing the gradient descent to "bounce" and 
slow down convergence. Optimizers like Adagrad and Adam protect against this problem by creating a separate effective 
learning rate per feature. But optimizers can’t prevent a wide range of values within a single feature; in those cases,
they must be normalized.  
