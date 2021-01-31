# Deep Neural Network Models:  
There are limitations to matrix factorisation that include:
* **Difficulty using side features, so any features beyond the queryID/itemID:** The model can only be queried with a user 
or item that is present in the training set.
* **Relevance of recommendations:**  Popular items are recommended to everyone, especially when using dot product as a 
similarity measure. It is better to capture specific user interests.  

DNNs can easily incorporate query features and item features (due to the flexibility of the input layer of the network), 
which can help capture the specific interests of a user and improve the relevance of recommendations.  

#### Softmax DNN for Recommendation:  
One possible DNN model is softmax, which treats the problem as a multiclass prediction problem in which:
* The input is the user query
* The output is a probability vector with size equal to the number of items in the corpus, representing the probability 
to interact with each item; for example, the probability to click on or watch a YouTube video.  

#### Input:    
The input to a DNN can include:
* Dense features (for example, watch time and time since last watch)
* Sparse features (for example, watch history and country)  

Unlike the matrix factorisation approach, side features such as age or country can be added. Denote the input 
vector by x:

![alt text](https://developers.google.com/machine-learning/recommendation/images/Inputlayer.svg
"Softmax")  

#### Model Architecture:  
The model architecture determines the complexity and expressivity of the model. By adding hidden layers and non-linear 
activation functions (for example, ReLU), the model can capture more complex relationships in the data. However, 
increasing the number of parameters also typically makes the model harder to train and more expensive to serve.  

#### Softmax Output: Predicted Probability Distribution:  
The model maps the output of the last layer, <img src="https://latex.codecogs.com/svg.latex?%5Cpsi%20%28x%29">, 
through a softmax layer to a probability distribution:  

<img src="https://latex.codecogs.com/svg.latex?%5Chat%7Bp%7D%3Dh%28%5Cpsi%20%28x%29V%5E%7BT%7D%29">  

where:  
* <img src="https://latex.codecogs.com/svg.latex?h%3A%5Cmathbb%7BR%7D%5E%7Bn%7D%5Crightarrow%20%5Cmathbb%7BR%7D%5E%7Bn%7D"> 
    is a softmax function given by:  
    
    <img src="https://latex.codecogs.com/svg.latex?h%28y%29_%7Bi%7D%3D%5Cfrac%7Be%5E%7By_%7Bi%7D%7D%7D%7B%5Csum%20_%7Bj%7De%5E%7By_%7Bi%7D%7D%7D"> 
        
* <img src="https://latex.codecogs.com/svg.latex?V%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%5Ctimes%20d%7D"> is the matrix of 
weights of the softmax layer  

The softmax layer maps a vector of scores <img src="https://latex.codecogs.com/svg.latex?y%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%7D"> 
also known as logits, to a probability distribution.

![alt text](https://developers.google.com/machine-learning/recommendation/images/Ppd.svg
"Softmax")  

#### Loss Function:  
Finally, define a loss function that compares the following:  
* The output of the softmax layer (a probability distribution)
* The ground truth, representing the items the user has interacted with (for example, YouTube videos the user clicked 
or watched). This can be represented as a normalized multi-hot distribution (a probability vector).

![alt text](https://developers.google.com/machine-learning/recommendation/images/LossFunction.svg
"Softmax")  

#### Loss Function:  
The log probability of an item is (up to an additive constant) the dot product of two d-dimensional vectors, 
which can be interpreted as query and item embeddings:
* The output of the last hidden layer. Called the embedding of the query
* The vector of weights connecting the last hidden layer to output. Called the embedding of item  

Items with the highest probability are the items with the highest dot product.

#### DNN and Matrix Factorisation:  
In both the softmax model and the matrix factorisation model, the system learns one embedding vector *V<sub>j</sub>* 
per item *j*. What was called the item embedding in matrix factorisation is now the matrix of weights of the softmax 
layer.  

The query embeddings are different. Instead of learning one embedding per query, the system learns a mapping from the 
query feature to an embedding. Therefore the DNN model is like a generalization of matrix factorisation in which you 
replace the query side by a nonlinear function.  

#### Item Features:  
Instead of learning one embedding per item, the model can learn a nonlinear function that maps item features to an 
embedding. A two-tower neural network, which consists of two neural networks can be used to achieve this:
* One neural network maps query features to query embedding
* One neural network maps item features to item embedding  

The output of the model can be defined as the dot product of query and item. This is not a softmax model anymore. The 
new model predicts one value per pair, instead of a probability vector for each query.

#### Softmax Training:  
The softmax training data consists of the query features and a vector of items the user interacted with (represented 
as a probability distribution). These are marked in blue in the following figure. The variables of the model are the 
weights in the different layers. These are marked as orange in the following figure. The model is typically trained 
using any variant of stochastic gradient descent.  

![alt text](https://developers.google.com/machine-learning/recommendation/images/Training.svg
"Softmax")  

#### Negative Sampling:  
Since the loss function compares two probability vectors (the ground truth and the output of the model), 
computing the gradient of the loss (for a single query) can be prohibitively expensive if the corpus size is too big.  

The system could be set up to compute gradients only on the positive items (items that are active in the ground truth 
vector). However, if the system only trains on positive pairs, the model may suffer from folding.  

Instead of using all items to compute the gradient (which can be too expensive) or using only positive items 
(which makes the model prone to folding), you can use negative sampling. More precisely, you compute an approximate 
gradient, using the following items:
* All positive items (the ones that appear in the target label)
* A sample of negative items  

There are different strategies for sampling negatives:
* Sample uniformly
* Give higher probability to items with higher score. These are examples that contribute the most to the gradient, 
often called hard negatives.

#### Folding:  
In the fugure below each colour represents a different category og queries and items. Queries (squares) only interact 
with items (circles) of the same colour. The model may learn how to place the query/item embeddings of a given colour 
relative to each other (correctly capturing similarity within that colour), but embeddings from different colours may 
end up in the same region of the embedding space, by chance. This is folding and can lead to false recommendations, the 
query may incorrectly predict a high score for an item from a different group.  

Negative examples can be used in training to show the model that embeddings from different groups should be pushed away 
from each other.  

![alt text](https://developers.google.com/machine-learning/recommendation/images/Negatives.svg
"Folding")  

#### Matrix Factorisation vs Softmax:  
DNN models solve many limitations of Matrix Factorisation, but are typically more expensive to train and query. There 
are positives and negatives of each method: 

Matrix Factorisation:
* **Query features:** Not easy to include.
* **Cold start:** Does not easily handle out-of vocab queries or items. Some heuristics can be used (for example, 
for a new query, average embeddings of similar queries).
* **Folding:** Folding can be easily reduced by adjusting the unobserved weight in WALS.
* **Training scalability:** Easily scalable to very large corpora (perhaps hundreds of millions items or more), but
 only if the input matrix is sparse.
* **Serving scalability:** Embeddings U, V are static, and a set of candidates can be pre-computed and stored.

Softmax DNN:
* **Query features:** Can be included.
* **Cold start:** Easily handles new queries.
* **Folding:** Prone to folding. Need to use techniques such as negative sampling or gravity.
* **Training scalability:** Harder to scale to very large corpora. Some techniques can be used, such as hashing, 
negative sampling, etc.
* **Serving scalability:** Item embeddings V are static and can be stored. However the query embedding usually needs 
to be computed at query time, making the model more expensive to serve.  

Matrix factorisation is usually the better choice for large corpora. It is easier to scale, cheaper to query, and less 
prone to folding.  

DNN models can better capture personalised preferences, but are harder to train and more expensive to query. DNN models 
are preferable to matrix factorisation for scoring because DNN models can use more features to better capture relevance. 
Also, it is usually acceptable for DNN models to fold, since you mostly care about ranking a pre-filtered set of 
candidates assumed to be relevant.
