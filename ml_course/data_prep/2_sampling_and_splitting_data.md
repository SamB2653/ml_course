# Sampling and Splitting Data: 
If there is too much data to be processed then a subset sample of the data can be taken for training. How the sample 
is determined depends on the problem, what is being predicted and what features are wanted. For example:
* To use the feature previous query, you need to sample at the session level, because sessions contain a sequence of queries.
* To use the feature user behavior from previous days, you need to sample at the user level. 

If the data includes PII (personally identifiable information), it may need to filtered from the data. The filtering of 
this data can skew the distribution as information in the tail (the part of the distribution with very low values, far 
from the mean) will be lost. It is useful to filter data from the tail as the data set will be biased towards the head 
queries. The model will do worse serving on examples from the tail, since those were values filtered out when training.

#### Imbalanced Data:
A classification data set with skewed class proportions is called imbalanced. Classes that make up a large proportion 
of the data set are called majority classes. Those that make up a smaller proportion are minority classes. Degree of 
imbalance:
* **Mild:** 20-40% of the data set (proportion of minority class)
* **Moderate:** 1-20% of the data set
* **Extreme:** <1% of the data set

A particular sampling technique may need to be applied if there is a classification task with an imbalanced data set. 
For example if a data set has 99.5% negative and 0.5% positive values in a classification problem then the model will 
spend most of its time on the negative values. A batch size of 128 will have many batches with no positive examples, 
therefore the gradients will be less informative.  

If there is an imbalanced data set try training on the true distribution. If the model works well and generalizes then 
the model works, if not, try the following downsampling and upweighting technique.

#### Downsampling and Upweighting:  
An effective way to handle imbalanced data is to downsample and upweight the majority class:
* **Downsampling:** Training on a disproportionately low subset of the majority class examples.
* **Upweighting:** Adding an example weight to the downsampled class equal to the factor by which was downsampled.  

For example for a data set with 1 positive for every 200 negatives. **Downsample the majority class**. The data can be 
downsampled by a factor of 20 to 1 in 10 being a positive value, 10% of the data is now positive so the model will 
train better.  

The next step is to **upweight the downsampled class**. Add example weights to the downsampled class, as the data was 
downsampled by a factor of 20, the weight will be 20. An example weight of 10 means the model treats the example as 10 
times as important (when computing loss) as it would an example of weight 1. The weight should equal the factor used to 
downsample:

*{example weight} = {original example weight} x {downsampling factor}*  

We were trying to make our model improve on the minority class so why would we upweight the majority?
* **Faster convergence:** During training, we see the minority class more often, which will help the model converge faster.
* **Disk space:** By consolidating the majority class into fewer examples with larger weights, we spend less disk space 
storing them. This savings allows more disk space for the minority class, so we can collect a greater number and a 
wider range of examples from that class.
* **Calibration:** Upweighting ensures our model is still calibrated; the outputs can still be interpreted as probabilities.

#### Data Splitting:  
After data collection and sampling if needed the next step is to split the data into training sets, validation sets, 
and testing sets.  

A pure random split is not always the right approach. It would be an issue when examples are naturally clustered into 
similar examples, a pure random split would split a cluster across sets, causing skew. A solution to this would be to 
split on time rather than randomly, this could preserve clusters. A split by time could work like this:
* Collect 30 days of data
* Train on data from Days 1-29
* Evaluate on data from Day 30  

For online systems, the training data is older than the serving data, so this technique ensures the validation set 
mirrors the lag between training and serving. However, time-based splits work best with very large data sets, such as 
those with tens of millions of examples. In projects with less data, the distributions end up quite different between 
training, validation, and testing. To design a split that is representative of the data, consider what the data 
represents. The golden rule applies to data splits as well: the testing task should match the production task as 
closely as possible.

#### Randomization:  
Make the data generation pipeline reproducible. For example when adding a new feature, data sets should be identical 
except for this new feature. Data generation runs should be reproducible. Therefore, ensure any randomization in data 
generation can be made deterministic:
* **Seed the random number generators:** Seeding ensures that the RNG outputs the same values in the same order each time 
it is run, recreating the data set.
* **Use invariant hash keys:** Hash each example, and use the resulting integer to decide in which split to place the 
example. The inputs to the hash function should not change each time the data generation program is run. Don't use the 
current time or a random number in the hash if there is a need to recreate the hashes on demand. Do not always include 
or always exclude as:
* The training data set will see a less diverse set of queries
* The evaluation sets will be artificially hard, because they won't overlap with the training data.
