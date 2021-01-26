# Collecting and Constructing the Data Set: 
The majority of time of a ML project is spent constructing data sets and transforming data. Before data transformation 
is started the data set must be constructed:
* Collection of raw data
* Identifying feature and label sources
* Selecting a sampling strategy
* Splitting the data

#### Size and Quality of a Data Set:
The model should train on at least an order of magnitude more examples than trainable parameters. Simple models with 
large data sets are better than complex models on small data sets. Linear regression can work very well on large 
data sets.  

There is no point having a large amount of poor data. A quality data set is one that lets you succeed with the business 
problem you are looking at, so if it accomplishes its intended task. Aspects of quality that tend to correspond with 
good performing models:
* **Reliability:** How well can the data be trusted? To measure this the following must be determined:  
    * How common are label errors?
    * Are the features noisy?
    * Is the data properly filtered for the specific problem?
    * The following can make the data unreliable:
        * Omitted values
        * Duplicate examples
        * Bad labels, eg the label is not the correct value when compared to the true answer
* **Feature Representation:** Mapping the data to useful features. Consider the following questions:
    * How is the data shown to the model?
    * Should numeric values be normalized?
    * How should outliers be handled?
* **Minimising Skew:** During training, use only the features that will be available in serving, and make sure 
the training set is representative of the serving traffic. Consider what data is available to the model at prediction 
time to prevent different results from offline and online models.

#### Data Logs:  
When assembling a training data set sources of data may have to be joined. There are various common types of input data: 
* **Transactional Logs:** Records a specific event, eg an IP address making a query and the date and time at which the 
query was made.
* **Attribute Data:** Contains snapshots of information that is not specific time. Snapshots of information could be:
    * User demographics
    * Search history at time of the query
* **Aggregate Statistics:** You can create a type of attribute data by aggregating several transactional logs, creating 
aggregate statistics. In this case, you can look at many transactional logs to create a single attribute for a user. 
Aggregate statistics create a single attribute for a user. For example: 
    * Frequency of user queries
    * Average click rate 

#### Joining Log Sources:  
Each type of log tends to be in a different location. When collecting data for the ML model, different sources may have 
to be joined together to create the data set. Some examples:  
* Leverage the user's ID and timestamp in transactional logs to look up user attributes at time of event.
* Use the transaction timestamp to select search history at time of query.  

#### Identifying Labels and Sources: 
The best label is a direct label of what is to be predicted. A derived label is where a user has taken an action that 
implies that they like something, this does not directly measure to what is being predicted but could be a reliable 
indicator. The derived label should have a strong connection with the desired prediction for a good output.  

The output of the model could be either an Event or an Attribute. This results in the following two types of labels:  
* **Direct label for Events:** Did the user click the top search result? Log the user behavior during the event for use 
as the label. When labeling events, ask the following questions:
    * How are the logs structured?
    * What is considered an event in the logs?
* **Direct label for Attributes:** Will the advertiser spend more than $X in the next week? use the previous days of 
data to predict what will happen in the subsequent days. Consider seasonality or cyclical effects, weekends could have 
an impact so use a 14 day window, or use years due to season effects.  

Direct labels need logs of past behavior. Machine learning makes predictions based on what has happened in the past, 
so cannot make predictions without historical data. If the product is new and/or there is no historical data:
* Use a heuristic for a first launch, then train a system based on logged data
* Use logs from a similar problem to bootstrap your system
* Use human raters to generate data by completing tasks
