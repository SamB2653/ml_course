# Problem Framing: 
It is important to outline the key features and outcomes of the model before it is created.

* **What should the model do?** The goal of the ML model and what problem it solves.
* **What is the ideal outcome?** Determine the desirable outcome of the model. This could be recommendations of a product.
* **How will success be determined?** Success and failure metrics should be phrased differently to evaluation metrics 
such as accuracy and recall. Specify anticipated outcomes to prevent sunk costs into the model. So a failure condition 
could be that the recommendations from a model are no better than the system currently in place.
* **Are the metrics measurable?** A measurable metric provides enough information for successful real-world evaluation. 
The following are useful for determining this:
    * How will the metrics be measured?
    * When will the metrics be measured?
    * How long will it take to determine if the model is a success or failure?
* **Determine non measurable failure scenarios:** If the model is always recommending the same content which will lead 
to a negative user experience.
* **What output should the model produce?** This could be a value, a label, cluster ect. For example you cannot tell if
a user liked content without asking the user, a proxy label (label not in the data set) should be used instead. So 
time played or money spent could be a proxy label for enjoyment. The output should also be connected to the ideal 
outcome so the output should be something important, the stronger the connection between the output labels and the true 
outcome, the more likely that the model is optimising the correct thing.
* **Can you obtain example outputs for the training data?** Supervised ML relies on labeled data, whereas unsupervised 
does not require labeled data. Output examples may need to be engineered.
* **How will the output be used?** The model cal make predictions in two ways:
    * **Online:** Real time, in response to user activity
    * **Offline:** As a batch and the cached  
* **How will the output be implemented?** Determine how the model output will be used within the product, eg:

    ```python3
    click_probability = call_model(user)
    if click_probability > 0.05:
        send_notification(user)
    ```

    This will turn the models predictions into decisions. Also the following should be looked into:
    * What data does the code have access to when calling the model? You can only train on features that you would also
     have access to when you call the model
    * What are the latency requirements? eg is the user waiting for a response?  
* **Avoid bad objectives:** ML systems are good at pursuing the objectives they're given, however they can also produce 
negative outcomes if the wrong objectives are chosen. The model can cause the user experience to suffer even if the 
model is a good one.
* **Does the problem need ML?** If the problem is severely time limited it could take too long to implement a ML model, 
also the problem could be over engineered with ML when a heuristic (non-ML solution) could be used.

#### Formulating a Problem:
A good approach for framing a ML problem is the following:  
* **Articulate the problem:** Which subtype of ML should be used?
    * How many categories? 2 = Binary Classification, >2 = Multi-class classification
    * For Multi-class classification: How many categories for a single example? 1 = Multi-class Single-label, >1 
    Multi-class Multi-label.
    * For regression: How many numbers are output? 1 = uni-dimensional regression, >1 = multidimensional regression
* **Start simple:** Start with the most simple model possible, one that is easy to understand. Once the pipeline has 
been created improve the model. This is a good baseline.
* **Identify data sources:** Provide the answers to these questions:
    * How much labeled data is there?
    * What is the data source?
    * Is the label closely connected to the decision that will be made?
* **Design data for the model:** Identify the data the ML system should use for the predictions (features and labels)
* **Determine where data comes from:** Consider how the data pipeline will be created. The example output should not 
be hard to obtain, otherwise another could be used.
* **Determine easily obtained inputs:** Start with few inputs (around 3) and see the initial outcome. It is useful to 
consider what inputs would be used in a non-ML solution for this. Start with a lightweight pipeline.
* **Ability to learn:** Find areas that may cause the model to have difficulties learning. Examples:
    * Data doesnt contain enough positive labels
    * Training data does not contain enough examples
    * Labels too noisy
    * System memorises training data but cannot generalise new data
* **Identify potential bias:** Determine sources of potential bias, such as training data that is not representative 
of the whole user base.
