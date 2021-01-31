# Recommendation Systems: 
Some key terms to recap:
* **Items:** the entities a system recommends.
* **Query:** the information a system uses to make recommendations, they can be a combination of the following:
    * User information (id of user, items interacted with ect).
    * Additional context (time of day, users device ect).
* **Embedding:** A mapping from a discrete set to a vector space. For recommendations this is the set of queries or 
set of items to recommend.

#### Common System Structure:  
A common architecture for a recommendation system consists of the following components:
* **Candidate generation:** The system starts from a large set of data and then generates a much smaller subset of data. If there are billions of
entries then they need to be reduced to thousands as the model needs to evaluate queries quickly.
* **Scoring:** Another model scores and ranks the candidates to select the set of items (order of 10) to display to the user. Since the 
model evaluates a small subset of items, the system can use a more precise model relying on additional queries.
* **Re-ranking:** The system takes into account the additional constraints for the final ranking, for example the system could remove 
items that the user explicitly disliked or add weights to content that is being pushed. This can prevent the model from 
becoming stale.

#### Candidate Generation Overview:  
Candidate generation is the first stage of recommendation. When given a query the system generates a set of relevant 
candidates, there are two common candidate generation approaches:  
* **Content-based filtering:** Uses similarity between items to recommend items similar to what the user likes. If user 
A plays content with theme B then the system will recommend content with theme B.
* **Collaborative filtering:** Uses similarities between queries and items simultaneously to provide recommendations. 
If user A is similar to user B, and user B plays content 1, then the system can recommend content 1 to user A.

#### Embedding Space:  
Both content-based and collaborative filtering map each item and each query (or context) to an embedding vector in a 
common embedding space:  

<img src="https://latex.codecogs.com/svg.latex?E%20%3D%20%5Cmathbb%7BR%7D%5E%7Bd%7D" title="Equation" />  

The embedding space is low-dimensional (d is muh smaller than the corpus), and captures some latent structure of the 
item or query set. Similar items are placed closer together within embedding space, closeness is defined by a similarity 
measure.

#### Similarity Measures:  
A similarity measure is a function that takes a pair of embeddings and returns a scalar that measures their similarity:

<img src="https://latex.codecogs.com/svg.latex?s%20%3A%20E%20%5Ctimes%20E%20%5Crightarrow%20%5Cmathbb%7BR%7D" title="Equation" />  

The embeddings can be used for candidate generation. For a query with the embedding <img src="https://latex.codecogs.com/svg.latex?q%20%5Cin%20E">
the system looks for item embeddings <img src="https://latex.codecogs.com/svg.latex?x%20%5Cin%20E">
that are close to *q*, embeddings with a high similarity <img src="https://latex.codecogs.com/svg.latex?s%28q%2Cx%29">.  

The following are commonly used to determine the degree of similarity:
* Cosine
* Dot product
* Euclidean distance

#### Cosine:  
The cosine angle between the two vectors:  

<img src="https://latex.codecogs.com/svg.latex?s%28q%2Cx%29%20%3D%20cos%28q%2Cx%29">

#### Dot Product:  
The dot product of two vectors is:  

<img src="https://latex.codecogs.com/svg.latex?s%28q%2Cx%29%20%3D%20%5Cleft%20%5Clangle%20q%2Cx%20%5Cright%20%5Crangle%20%3D%20%5Csum%20_%7Bi%3D1%7D%5E%7Bd%7Dq_%7Bi%7Dx_%7Bi%7D">  

Or given by the cosine of the angle multiplied by the product of the norms:  

<img src="https://latex.codecogs.com/svg.latex?s%28q%2Cx%29%20%3D%20%5Cleft%20%5C%7C%20x%20%5Cright%20%5C%7C%5Cleft%20%5C%7C%20q%20%5Cright%20%5C%7Ccos%28q%2Cx%29">  

If the embeddings are normalised then the dot product and the cosine coincide.

#### Euclidean distance:  
The usual distance in Euclidean space where a smaller distance means a higher similarity. 

<img src="https://latex.codecogs.com/svg.latex?s%28q%2Cx%29%20%3D%20%5Cleft%20%5C%7C%20q-x%20%5Cright%20%5C%7C%20%3D%20%5Cleft%20%5B%20%5Csum%20_%7Bi%3D1%7D%5E%7Bd%7D%20%28q_%7Bi%7D-x_%7Bi%7D%29%5E%7B2%7D%20%5Cright%20%5D%5E%7B%5Cfrac%7B1%7D%7B2%7D%7D">  

When the embeddings are normalised the squared Euclidean distance coincides with dot product and cosine up to a constant,
since in that case:  

<img src="https://latex.codecogs.com/svg.latex?%7B%5Cfrac%7B1%7D%7B2%7D%7D%5Cleft%20%5C%7C%20q-x%20%5Cright%20%5C%7C%5E%7B2%7D%20%3D%201%20-%20%5Cleft%20%5Clangle%20q%2Cx%20%5Cright%20%5Crangle">  

#### Comparing Similarity Measures:  
Compared to the cosine, the dot product similarity is sensitive to the norm of the embedding. The larger the norm of an 
embedding, the higher the similarity (for items with acute angle) and the more likely the item is to be recommended. This 
has the following impacts:
* **Items that appear very frequently in the training set tend to have embeddings with large norms**. If capturing 
popularity information is desirable, then the dot product should be used. However the popular items may end up dominating 
the recommendations. Less emphasis can be put on the norm of the item, for example (dot product with constant):  

    <img src="https://latex.codecogs.com/svg.latex?s%28q%2Cx%29%3D%5Cleft%20%5C%7C%20q%5Cleft%20%5C%7C%5E%7B%5Calpha%20%7D%20%5Cright%20%5C%7C%20x%5Cright%20%5C%7C%5E%7B%5Calpha%20%7Dcos%28q%2Cx%29">  

    where <img src="https://latex.codecogs.com/svg.latex?%5Calpha%20%5Cin%20%280%2C1%29">

* **Items that appear very rarely may not be updated frequently during training**. If they are initialized with a large 
norm, the system may recommend rare items over more relevant items. To avoid this an appropriate regularisation as well 
embedding initialisation should be used.
