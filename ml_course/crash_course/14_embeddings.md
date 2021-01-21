#Embeddings:  
An embedding is a relatively low-dimensional space into which you can translate high-dimensional vectors. Embeddings 
make it easier to do machine learning on large inputs, such as sparse vectors. An embedding should capture some of the 
semantics of the input by placing semantically similar inputs close together in the embedding space. An embedding 
can be learned and reused across models.

####Motivation From Collaborative Filtering:  
Collaborative filtering is where predictions are made about the interests of a user based on interests of many other 
users. **Example:** A good example of this is movie recommendations, if we had 1,000,000 users with a catalog of 
500,000 movies we would want to recommend a movie to each user. Embeddings are used to determine how similar each
movie is to each other, movies will be embedded in low dimensional space where similar movies are located close to
each other.

####Categorical Input Data:  
Categorical data refers to input features that represent finite items from a set of choices. For example it could be
a set of movies a user has watched or a set of words in a document ect. Categorical data is most efficiently represented
via sparse tensors, which are tensors with a low amount of non-zero elements. **Example:** a unique ID could be assigned
to each move and represent each user by a sparse tensor of each movie they have watched. A tensor would look like this:

* [1, 5, 678]  

When looking at the example where words are used a large input layer could be used with a node for every word that is
being looked at. **Example:** if there are 500,000 unique words within the data you could represent a word with a vector
that is 500,000 long. So a word such as car would have a 1 in the 13,252nd position and 0 in the rest:

* [0, ... , 1, ... , 0]

This is one hot encoding. The size of the vector will cause problems within the model due to the one node per word 
method that would be used. Sparse representations can be hard to learn.

####Size of Network:  
Huge input vectors cause a massive number of weights for a neural network. If you have M words in the vocabulary and N
nodes in the first layer of the network above the input, then you have NxM weights to train for in that layer. There
are also additional problems:

* **Amount of data:** The more weights in your model, the more data you need to train effectively.
* **Amount of computation:** The more weights, the more computation required to train and use the model.

####Lack of Meaningful Relations Between Vectors:  
You should consider if it makes sense to have "close" values between vectors of items within the model. **Example:**
A good use would be for looking at the relation between pixel values of RGB channels, where reddish blue is close to pure
blue, semantically and in terms of geometric distance between vectors. However for the words example a vector with 
index position 1200 "Dog" is not closer to a a vector in position 50,000 "Cat" than "Car" at index position 200.  

This is solved by using embeddings. These translate large sparse vectors to lower dimensional space that preserves the
semantic relationships.

####Translating to a Lower-Dimensional Space:  
High dimensional data, sparse data from the input can be mapped to lower dimensional space. See below the geometrical 
relationships between various words:

![alt text](https://developers.google.com/machine-learning/crash-course/images/linear-relationships.svg "embeddings")

Patterns can be detected from this and will help the learning task of the model.  

The embedding space should also be
small enough to reduce the computational power that is needed for the model, speeding up training. Even if an embedding
is hundreds of thousands of dimensions, this will be several orders of magnitude smaller than the inputs for the words
example stated earlier.  

####Embeddings as lookup tables:  
An embedding is a matrix where each column is the vector that corresponds to an item within your vocabulary, to get the
dense vector for a single item the column for that item will be retrieved. However with a multiple words (bag of words)
you can retrieve the embedding for each individual item and then add them together. If the sparse vector contains counts
of the vocabulary items, you can multiply each embedding by the count of its corresponding item before adding it to the 
sum.

####Embedding lookup as matrix multiplication:  
The lookup, multiplication, and addition procedure we've just described is equivalent to matrix multiplication. Given 
a 1 X N sparse representation S and an N X M embedding table E, the matrix multiplication S X E gives you the 1 X M 
dense vector.
