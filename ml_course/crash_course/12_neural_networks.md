#Neural Networks:  
Neural networks can be used to solve much more complex non linear problems.
####Structure:
Each input node on the input layer represents an input feature. Each node in the Hidden layer is the weighted sum of 
the input node values and the output is the weighted sum of the hidden layer nodes.

![alt text](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/example_network.svg
"Neural Networks Structure")

####Activation Functions:
To model a nonlinear problem, we can directly introduce a nonlinearity. We can pipe each hidden layer node through a 
nonlinear function. This is the activation function. So each value in the hidden layer is transformed by a
nonlinear function before being passed onto the weighted sums of the next layer.  

Nonlinear layers can be stacked onto other nonlinear layers to solve very complicated relationships between the inputs
and predicted outputs. Each added layer will add more complexity to the network. Each layer is effectively learning a 
more complex, higher-level function over the raw inputs.

####Common Activation Functions:
Two common activation functions are sigmoid and ReLU.
* Sigmoid activation function converts the weighted sum to a value between 0 and 1.
  
* <img src="https://latex.codecogs.com/gif.latex?F(x)=\frac{1}{e^{-x}}" title="F(x)=\frac{1}{e^{-x}}" />  

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png 
"Sigmoid")

* Rectified Linear Unit (ReLU) works better and is easier to compute than a sigmoid function.

* <img src="https://latex.codecogs.com/gif.latex?F(x)=&space;max(0,x)" title="F(x)= max(0,x)" />  

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/1920px-Activation_rectified_linear.svg.png
"ReLU")


Additionally any mathematical function can serve as an activation function. If σ represents an activation function then
the value of a node in the network is given by:

<img src="https://latex.codecogs.com/gif.latex?\sigma(w&space;\cdot&space;x&space;&plus;&space;b)" title="\sigma(w \cdot x + b)" />

####Neurons:
Not having enough neurons in a layer will underfit the model, however if you add too many neurons the model could start
memorizing the data, this will cause overfitting. Usually a more simple model will out perform a very complex model
due to this.  

It is also important to select the correct activation function for the output that you want, for example if you select
a non linear activation function then it will not be able to learn any nonlinearities, meaning the loss will be high.

####Continuous Visualization of Layers (low-dimensional neural network):
The following network classifies two spirals that are slightly entangled, using four hidden layers. Over time, we can 
see it shift from the “raw” representation to higher level ones it has learned in order to classify the data. While 
the spirals are originally entangled, by the end they are linearly separable.

![](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/img/spiral.1-2.2-2-2-2-2-2.gif)

####neural_network_simple.py:
Creating a simple neural network.
* Create a simple deep neural network.
* Tune the hyperparameters for a simple deep neural network.

####Training Neural Networks:
Backpropagation is the most common training algorithm for neural networks. It makes gradient descent feasible for
multi-layer neural networks. TensorFlow handles backpropagation automatically, so you don't need a deep understanding 
of the algorithm.  

The backpropagation algorithm decides how much to update each weight of the network after comparing the predicted 
output with the desired output for a particular example. How the error changes with respect to each weight is calculated
so the error derivative is used to update the weights.

####Best Practices - Failure Cases for Backpropagation:
**Vanishing Gradient:** The gradients for lower layers (layers closer to the input) can become very small. In deep
networks computing these gradients can involve taking the product of many small terms. When gradients tend towards 0 
for the lower layers then these layers train very slowly or not at all. **RelU activation function** can prevent this.  

**Vanishing Gradient:** If the weights in a network are very large then the gradients in lower layers involve products
of many large terms. With exploding gradients the gradients get too large to converge. **Batch normalization** and 
lowering the **learning rate** can prevent this.  

**Dead ReLU Units:** If the weighted sum for a ReLU unit falls below 0, the ReLU unit can get stuck. It outputs 0
activation, contributing nothing to the networks output and the gradients can no longer flow through it during 
backpropagation. With a source of gradients cut off, the input to the ReLU may not ever change enough to bring the
weighted sum back above 0. Lowering the **learning rate** can prevent RelU units from dying.  

####Dropout Regularization:
This works by randomly dropping out unit activations in a network for a single gradient step. The more you drop out, 
the stronger the regularization:
* 0.0 = no dropout regularization.
* 1.0 = dropout everything (model learns nothing).
* Pick a value between 0.0 and 1.0.
