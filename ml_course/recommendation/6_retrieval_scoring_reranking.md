# Retrieval, Scoring and Re-Ranking:  

#### Retrieval:  
Given a query the following is done at the time the recommendation is made:
* For a matrix factorization model, the query (or user) embedding is known statically, and the system can look 
it up from the user embedding matrix.
* For a DNN model, the system computes the query embedding at serve time by running the network on the feature vector.  

Once the system has the query embedding the closest item embeddings are searched for within the embedding space. This 
is now a nearest neighbours problem. So return top k items according to the similarity score.  

For large scale retrieval, the system can calculate every candidates nearest neighbours in embedding space. This 
exhaustive scoring can be expensive for large corpora, but other strategies can be used instead:  
* If the query embedding is known statically, the system can perform exhaustive scoring offline, pre-computing and 
storing a list of the top candidates for each query. This is a common practice for related-item recommendation.
* Use approximate nearest neighbors.  

#### Scoring:  
After candidate generation, another model scores and ranks the generated candidates to select the set of items to 
display. The recommendation system may have multiple candidate generators that use different sources, such as the 
following:  
* Related items from a matrix factorization model
* User features that account for personalization
* "Local" vs "distant" items; that is, taking geographic information into account
* Popular or trending items
* A social graph; that is, items liked or recommended by friends  

The system combines these different sources into a common pool of candidates that are then scored by a single model and 
ranked according to that score. For example, the system can train a model to predict the probability of a user watching 
a video on YouTube given the following:
* Query features (for example, user watch history, language, country, time)
* Video features (for example, title, tags, video embedding)  

The system can then rank the videos in the pool of candidates according to the prediction of the model.

Candidate generators compute a score (such as the similarity measure in the embedding space). This should not be used 
for ranking as:
* Some systems rely on multiple candidate generators. The scores of these different generators might not be comparable.
* With a smaller pool of candidates, the system can afford to use more features and a more complex model that may 
better capture context.  

The choice of scoring function can dramatically affect the ranking of items, and ultimately the quality of the 
recommendations. For example with YouTube videos:
* Maximising click rate could recommend click bait videos
* Maximising watch time could recommend very long videos
* Increase diversity and maximise session watch time would recommend shorter videos but ones that engage the user  

Scoring can have positional bias. When serving the recommendations some will appear on the screen in lower positions, 
this will make them less likely to be clicked/played. The solution to this is to:
* Create position independent rankings
* Rank all candidates as if they are in the top on screen position if applicable  

#### Re-ranking:  
The system can re-rank the candidates to consider additional criteria or constraints. One re-ranking approach is to 
use filters that remove some candidates. Examples with videos: 
* Train a separate model to detect click bait videos and remove these from the main model  
* Manually transform the score returned by the ranker. So re-rank by modifying the score as a function of video age or 
video length for example  

The recommendations should always be as fresh as possible, incorporating the latest data within them. Keeping the 
recommendations fresh can be done in a few ways, such as:  
* **Re-running the training as regularly as possible**. Warm starting the training so the model doesnt have to learn 
everything from scratch. In Matrix factorisation the embeddings in the previous instance of the model would be kept.  
* **Create an "average" user to represent new users within matrix factorisation models**. The same embedding for each 
user is not needed as clusters based on user features can be created.  
* **Use a DNN such as Softmax or two-tower model**. This can be run on a query or item not seen in the training as it 
runs on feature vectors as inputs.  
* **Add document age as a feature**. For example how long the content has been out for.  

If the system always recommend items that are "closest" to the query embedding, the candidates tend to be very similar 
to each other. This lack of diversity can cause a bad or boring user experience. The solutions for this are:  
* Train multiple candidate generators using different sources
* Train multiple rankers using different objective functions
* Re-rank items based on genre or other metadata to ensure diversity  

Also ensure that the model does not have unconscious bias learned from the training data. The solutions to this fairness 
issue is:  
* Include diverse perspectives in design and development.
* Train ML models on comprehensive data sets. Add auxiliary data when your data is too sparse (for example, when 
certain categories are under-represented).
* Track metrics (for example, accuracy and absolute error) on each demographic to watch for biases.
* Make separate models for underserved groups.

#### softmax_model.py:  
Softmax recommendation system example with visualisations.
