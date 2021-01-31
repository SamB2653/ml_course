# Matrix Factorisation:  
Matrix factorisation is a simple embedding model. Given the feedback matrix:  

<img src="https://latex.codecogs.com/svg.latex?A%20%5Cin%20R%5E%7Bm%5Ctimes%20n%7D">  

where *m* is the number of users (queries) and *n* is the number of items the model learns:
* A user embedding matrix <img src="https://latex.codecogs.com/svg.latex?U%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20d%7D"> 
where row *i* is the embedding for user *i*.  
* An item embedding matrix <img src="https://latex.codecogs.com/svg.latex?V%20%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%5Ctimes%20d%7D"> 
where row *j* is the embedding for item *j*.  

The embeddings are learned such that the product *UV<sup>T</sup>* is a good approximation of the feedback matrix A. 
Observe that the *(i,j)* entry of the *U.V<sup>T</sup>* is simply the dot product <img src="https://latex.codecogs.com/svg.latex?%5Cleft%20%5Clangle%20U_%7Bi%7D%2C%20V_%7Bj%7D%20%5Cright%20%5Crangle"> 
of the embeddings of user *i* and item *j*, which you want to be close to *A<sub>i,j</sub>*.

Matrix factorisation typically gives a more compact representation than learning the full matrix. The full matrix has 
*O(nm)* entries while the embedding matrices *U*, *V* have *O((n+m)d)* entries, where the embedding dimension *d* is 
typically much smaller than *m* and *n*. Therefore matrix factorization finds latent structure in the data, assuming 
that observations lie close to a low-dimensional subspace.

#### Choosing the Objective Function:  
One intuitive objective function is the squared distance. To do this, minimize the sum of squared errors over all pairs 
of observed entries:

<img src="https://latex.codecogs.com/svg.latex?min_%7BU%5Cin%20%5Cmathbb%7BR%7D%5E%7Bm%5Ctimes%20d%7D%2C%20V%5Cin%20%5Cmathbb%7BR%7D%5E%7Bn%5Ctimes%20d%7D%7D%20%5Csum_%7B%28i%2Cj%29%5Cin%20obs%7D%5E%7B%7D%28A_%7Bij%7D-%5Cleft%20%5Clangle%20U_%7Bi%7D%2CV_%7Bj%7D%20%5Cright%20%5Crangle%29%5E%7B2%7D">  

In this objective function, you only sum over observed pairs (i, j), that is, over non-zero values in the feedback 
matrix. However, only summing over values of one is not a good idea. A matrix of all ones will have a minimal loss and 
produce a model that can't make effective recommendations and that generalizes poorly.  


![alt text](https://developers.google.com/machine-learning/recommendation/images/UnobservedEntries.svg
"CF")  

**Singular Value Decomposition (SVD)** is not a great solution, as in real applications, the matrix *A* may be very sparse.
The solution *UV<sup>T</sup>* (which corresponds to the model's approximation of the input matrix) will likely be close to zero, 
leading to poor generalisation performance.  

**Weighted Matrix Factorization** decomposes the objective into the following two sums:
* A sum over observed entries.
* A sum over unobserved entries (treated as zeroes).  

<img src="https://latex.codecogs.com/svg.latex?%5Csum_%7B%28i%2Cj%29%5Cin%20obs%7D%5E%7B%7Dw_%7Bi%2Cj%7D%28A_%7Bi%2Cj%7D-%5Cleft%20%5Clangle%20U_%7Bi%7D%2CV_%7Bj%7D%20%5Cright%20%5Crangle%29%5E%7B2%7D%20&plus;%20w_%7B0%7D%5Csum_%7B%28i%2Cj%29%5Cin%20obs%7D%5E%7B%7D%5Cleft%20%5Clangle%20U_%7Bi%7D%2CV_%7Bj%7D%20%5Cright%20%5Crangle%5E%7B2%7D">  

where *w<sub>i,j</sub>* is a function of the frequency of query *i* and item *j*.  

#### Minimizing the Objective Function:  
Common algorithms to minimize the objective function include:  
* **Stochastic gradient descent (SGD):** Generic method to minimize loss functions.
* **Weighted Alternating Least Squares (WALS):**  Specialized to this particular objective.

The objective is quadratic in each of the two matrices U and V. (Note, however, that the problem is not jointly convex. 
WALS works by initializing the embeddings randomly, then alternating between:
* Fixing U and solving for V.
* Fixing V and solving for U.  

Each stage can be solved exactly (via solution of a linear system) and can be distributed. This technique is guaranteed 
to converge because each step is guaranteed to decrease the loss. SGD and WALS have advantages and disadvantages:  

**SGD:**
* Very flexible, can use other loss functions
* Can be parallelised.
* Slower, does not converge as quickly
* Harder to handle the unobserved entries (need to use negative sampling or gravity)

**WALS:**
* Reliant on Loss Squares only
* Can be parallelised
* Converges faster than SGD
* Easier to handle unobserved entries
