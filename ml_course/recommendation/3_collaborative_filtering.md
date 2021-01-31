# Collaborative Filtering: 
Collaborative filtering uses similarities between users and items simultaneously to provide recommendations. The model 
can recommend an item to user A based on the interests of a similar user B. The embeddings can also be learned 
automatically, with no hand-engineering of features.

#### Recommendation Example:  
A recommendation system could have training data that consists of a feedback matrix where:
* Each row represents a user
* Each column represents an item (content)  

The content feedback consists of two categories:  
* **Explicit:** Users specify how much they liked a piece of content by providing a numerical rating 
* **Implicit:** If the user spends time on the content it is implied that they are interested  

When the user is served the recommendation, the recommendations should be based on:
* Similarity to content the user has liked in the past  
* Content that similar users liked  

#### Embedding:  
If each content was assigned a scalar in *[1,-1]* that describes a feature, then similar content could be placed on a 
one dimensional scale where content that is similar according to this one feature are grouped close together. The product 
of the content and user embedding should be closer to one for content that it is expected for the user to like.  

A single feature does not fully explain the users preferences. If another feature was added then it would produce a 
two dimensional embedding. Users would be grouped close together within this 2D space if they shared similar interests. 

In practice, the embeddings can be learned automatically, which is the power of collaborative filtering models. The 
model can learn an embedding vector for the users to best explain their preferences. Embeddings of users with similar 
preferences will be close together.

#### Collaborative Filtering Advantages and Disadvantages:  
**Advantages:**  
* **No domain knowledge necessary:** Embeddings are automatically learned.
* **Serendipity:** The model can help users discover new interests. In isolation, the ML system may not know the user is 
interested in a given item, but the model might still recommend it because similar users are interested in that item.
* **Good starting point:**  The system needs only the feedback matrix to train a matrix factorisation model. The 
system does not need contextual features. In practice, this can be used as one of multiple candidate generators.  

**Disadvantages:**  
* **Cannot handle fresh items:** The prediction of the model for a given (user, item) pair is the dot product of the 
corresponding embeddings. So, if an item is not seen during training, the system can't create an embedding for it and 
can't query the model with this item. This issue is often called the cold-start problem. However, the following 
techniques can address the cold-start problem to some extent:
    * **Projection in WALS:** Given a new item *i<sub>0</sub>* not seen in training, if the system has a few 
    interactions with users, then the system can easily compute an embedding *v<sub>i<sub>0</sub></sub>*  for this item 
    without having to retrain the whole model. The system simply has to solve the following equation or the weighted 
    version:  
    
        <img src="https://latex.codecogs.com/svg.latex?min_%7Bv_%7Bi_%7B0%7D%7D%5Cin%20%5Cmathbb%7BR%7D%5E%7Bd%7D%7D%5Cleft%20%5C%7C%20A_%7Bi_%7B0%7D%7D-Uv_%7Bi_%7B0%7D%7D%20%5Cright%20%5C%7C">  
    
    The preceding equation corresponds to one iteration in WALS: the user embeddings are kept fixed, and the system 
    solves for the embedding of item *i<sub>0</sub>*. The same can be done for a new user.  
    
    * **Heuristics to generate embeddings of fresh items:** If the system does not have interactions, the system can 
    approximate its embedding by averaging the embeddings of items from the same category.

* **Hard to include side features for query/item:** Side features are any features beyond the query or item ID. The 
side features might include country or age. Including available side features improves the quality of the model. 
Although it may not be easy to include side features in WALS, a generalization of WALS makes this possible.  

To generalize WALS, **augment the input matrix with features** by defining a block matrix <img src="https://latex.codecogs.com/svg.latex?%5Cbar%7BA%7D"> 
where:
* Block (0, 0) is the original feedback matrix
* Block (0, 1) is a multi-hot encoding of the user features
* Block (1, 0) is a multi-hot encoding of the item features  

Block (1, 1) is typically left empty. If you apply matrix factorization to <img src="https://latex.codecogs.com/svg.latex?%5Cbar%7BA%7D"> 
then the system learns embeddings for side features, in addition to user and item embeddings.

#### collaborative_filtering.py:  
Use the MovieLens data set to build a movie recommendation system:
* Exploring the MovieLens Data
* Matrix factorization using SGD
* Embedding Visualization
* Regularization in Matrix Factorization
