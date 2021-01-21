#Production ML Systems:  
There are a number of various components that are built around a ML model.

####Static vs Dynamic Training:  
There are in general 2 ways to train a model:

* **Static:** trained offline, model is trained once and then used for a while.
    * Easy to build and test, batch train and testing until the model is good
    * Requires monitoring of inputs
    * The model will grow stale as new data comes in, model is out of date quickly
* **Dynamic:** trained online, data is continually entered into the system and that data is used in the model.  
    * Continuously feed in training data so the model updates over time
    * Uses progressive validation instead of batch training and test
    * Needs constant monitoring, rollback and data quarantine capabilities
    * Adapts to changes so the model becoming stale is not an issue 

####Static vs Dynamic Inference: 
Inference is making predictions.

* **Offline:** all possible predictions in a batch are made, these are then written to a SSTable or Bigtable and then 
fed to a lookup table.
    * Don't need to worry about the cost of inference
    * Can likely use batch quota
    * Can do post-verification predictions on data before pushing to live
    * Can only predict things we know about, bad for long tail
    * Update frequency will be hours or days
* **Online:** predict on demand via a server.
    * Predict live data, good for long tail
    * Can be expensive computationally, latency sensitive to ms so can limit the model complexity 
    * Monitoring the actual model predictions as well as server components

####Data Dependencies:  
Input data (features) determine the models behaviour, the input features must be tested. Features included should be
useful:

* **Reliability:** What happens when the data source is not available? There should be a fallback feature for this
* **Visioning:** Does the system that computes the signal change? If it does change then the effect of this should be 
determined. A version number will determine if the signal has changed.
* **Necessity:** Does the usefulness of the signal justify the cost of adding it? Diminishing returns when adding new 
features.
* **Correlations:** Are any of the input signals tied together? Analysis could be needed to determine this.
* **Feedback loops:** Is the input depending on the output of the model?
