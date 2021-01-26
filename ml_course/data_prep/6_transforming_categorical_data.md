# Transforming Categorical Data: 
Some features may be discrete values that are not in an ordered relationship. This could be breeds of dogs ect. 
Categorical values can be represented as strings or even numbers, but you cannot compare these numbers or 
subtract them from each other.  

Often features that contain integer values should be represented as categorical data instead of as numerical data. 
For example, If you mistakenly represent a feature such as phone numbers numerically, then you're asking the model to 
find a numeric relationship between different phone numbers. Categorical representation will cause the model to 
separate the numbers and find separate signals for each number. See the indexed categorical data for when the total 
categories are small, there is a unique feature for each category. This is called a vocabulary.

![alt text](https://developers.google.com/machine-learning/data-prep/images/categorical-netview-indexed.svg
"Categorical Data (Indexed)")  

#### Vocabulary:
In a vocabulary, each value represents a unique feature. The model looks up the index from the string, assigning 1.0 to 
the corresponding slot in the feature vector and 0.0 to all the other slots in the feature vector.

![alt text](https://developers.google.com/machine-learning/data-prep/images/vocabulary-index-sparse-feature.svg
"Vocabulary Index")  

If your categories are the days of the week, you might, for example, end up representing Friday with the feature vector 
[0, 0, 0, 0, 1, 0, 0]. However, most implementations of ML systems will represent this vector in memory with a sparse 
representation. A common representation is a list of non-empty values and their corresponding indices, for example, 
1.0 for the value and [4] for the index. This allows you to spend less memory storing a huge amount of 0s and allows 
more efficient matrix multiplication. In terms of the underlying math, the [4] is equivalent to [0, 0, 0, 0, 1, 0, 0].  

Out of Vocab (OOV) can be used to catch very rare categories so the model doesnt waste time on training for rare 
categories. This is a catch all category.  

#### Hashing:
Another option is to hash every string (category) into your available index space. Hashing often causes collisions, 
but you rely on the model learning some shared representation of the categories in the same index that works well for 
the given problem.  

For important terms, hashing can be worse than selecting a vocabulary, because of collisions. On the other hand, 
ashing doesn't require you to assemble a vocabulary, which is advantageous if the feature distribution changes heavily 
over time.

![alt text](https://developers.google.com/machine-learning/data-prep/images/vocab-hash-string.svg
"Hashing")  

#### Hybrid of Hashing and Vocabulary:  
You can take a hybrid approach and combine hashing with a vocabulary. Use a vocabulary for the most important categories 
in your data, but replace the OOV bucket with multiple OOV buckets, and use hashing to assign categories to buckets.

The categories in the hash buckets must share an index, and the model likely won't make good predictions, but we have 
allocated some amount of memory to attempt to learn the categories outside of our vocabulary.

![alt text](https://developers.google.com/machine-learning/data-prep/images/vocab-hybrid.svg
"Hashing Vocab Hybrid")  

#### Embeddings:
an embedding is a categorical feature represented as a continuous-valued feature. Deep models frequently convert the 
indices from an index to an embedding.  Since embeddings are trained, they're not a typical data transformation, they 
are part of the model. They're trained with other model weights, and functionally are equivalent to a layer of weights. 
They therefore are not stored on disk like the previous methods can be.


![alt text](https://developers.google.com/machine-learning/data-prep/images/vocabulary-index-sparse-feature-embedding.svg
"Embeddings")  
