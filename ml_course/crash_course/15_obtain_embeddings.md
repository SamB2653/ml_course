# Obtaining Embeddings:  
When learning a d-dimensional embedding each item is mapped to a point in a d-dimensional space so that the similar 
items are nearby in this space. See below the embedding weights and the geometric view. The weights between an input 
node and the nodes in the d-dimensional embedding layer correspond to the coordinate values for each of the d axes.

![alt text](https://developers.google.com/machine-learning/crash-course/images/dnn-to-geometric-view.svg "embeddings")

#### Standard Dimensionality Reduction Techniques:  
There are many existing mathematical techniques for capturing the important structure of a high-dimensional space in a 
low dimensional space. In theory, any of these techniques could be used to create an embedding for a machine learning 
system.  

For example, principal component analysis (PCA) has been used to create word embeddings. Given a set of instances like 
bag of words vectors, PCA tries to find highly correlated dimensions that can be collapsed into a single dimension.

#### Training an Embedding as Part of a Larger Model:  
You can also learn an embedding as part of the neural network for your target task. This approach gets you an embedding 
well customized for your particular system, but may take longer than training the embedding separately.  

![alt text](https://developers.google.com/machine-learning/crash-course/images/EmbeddingExample3-1.svg "embeddings")

In general, when you have sparse data (or dense data that you'd like to embed), you can create an embedding unit that 
is just a special type of hidden unit of size d. This embedding layer can be combined with any other features and hidden 
layers. As in any deep neural network, the final layer will be the loss that is being optimized. 
 
**Example:** performing collaborative filtering, the goal is to predict a user's interests from the interests of other users.
This can be modeled as a supervised learning problem by randomly setting aside (or holding out) a small number of 
the movies that the user has watched as the positive labels, and then optimize a softmax loss.  

**Example:** When creating an embedding later of for the words in a housing ad you'd optimize the L<sub>2</sub> loss 
using the known house prices of houses within the training data as the label.
