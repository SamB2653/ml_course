# Problem Framing - Key Terms: 
Workflow of problem framing (defining a problem and proposing a solution):
* Articulate the problem
* See if labeled data exists  
* Design data for the model  
* Determine where data comes from  
* Determine easily obtained inputs  
* Determine quantifiable outputs

#### Key Terms Recap:
Supervised learning is where a model is provided with labeled training data (labeled data is the "answer" to the problem).
Features of the data are the inputs of the data that are used to make the predictions. Supervised learning uses the 
features with their corresponding labels and determines a relationship using whichever algorithm is suited to the 
problem (creating a model).
  
Unsupervised learning an unlabeled data set is used, this means the model must infer its own rules from the features of
the data to identify patterns. For example clustering a data set could be used for categorising data, once the boundaries 
have been determined, new data can easily be categorised.  

Reinforcement learning is different to the previous two mentioned methods as the model will be told not to get to the 
fail condition, when the model is not failing it receives a reward (reward function), therefore improving performance. 
This type of model is much less stable than the previous two mentioned as it can be difficult to design a good 
reward function. Also creating the environment for the model to interact with the environment (eg a game) can be very 
difficult.

#### Types of Problems:
Common subclasses of ML problems:  
* **Classification:** Pick one of N labels
* **Regression:** Predict numerical values
* **Clustering:** group similar examples
* **Association Rule Learning:** Infer likely association patterns in data (recommendations)
* **Structured Output:**  create complex output (eg image recognition)
* **Ranking:** Identify position on a scale or status 
