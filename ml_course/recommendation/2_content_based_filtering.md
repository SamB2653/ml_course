# Content-Based Filtering: 
Content-based filtering uses item features to recommend other items similar to what the user likes, based on their 
previous actions or explicit feedback.

The model should recommend items relevant to each user. Firstly a similarity metric should be picked, such as dot-product. 
Then the system should be set up to score each candidate item according to the similarity metric. The model is specific 
to the user as no other user information is used. Below is a feature matrix where each row represents a content item and 
each column represents a feature:

![alt text](https://developers.google.com/machine-learning/recommendation/images/Matrix1.svg
"Content-Based Filtering")  

#### Using Dot Product as a Similarity Measure:  
Consider a case when the user embedding *x* and the content embedding *y* are both binary vectors. Since:  

<img src="https://latex.codecogs.com/svg.latex?%5Cleft%20%5Clangle%20x%2Cy%20%5Cright%20%5Crangle%20%3D%20%5Csum%20_%7Bi%3D1%7D%5E%7Bd%7Dx_%7Bi%7Dy_%7Bi%7D">  

a feature appearing in both *x* and *y* contributes a 1 to the sum, so  <img src="https://latex.codecogs.com/svg.latex?%5Cleft%20%5Clangle%20x%2Cy%20%5Cright%20%5Crangle"> 
is the number of features that are active in both vectors simultaneously. A high dot product indicates more common 
features, so a higher similarity.  

#### Content-Based Filtering Advantages and Disadvantages:  
**Advantages:**  
* The model doesnt need any data about users as the recommendations are specific to each user. This makes it easier to 
scale for large amounts of users.  
* The model can capture specific interest of a user and can recommend niche items that very few other users are 
interested in.  

**Disadvantages:**  
* Feature representation of the items are hand-engineered in many cases so requires a lot of specific domain knowledge. 
The model is only as good as the hand-engineered features.  
* The model can only make recommendations based on the existing interests of the user. So the model has a limited 
ability to expand on the users existing interests.
