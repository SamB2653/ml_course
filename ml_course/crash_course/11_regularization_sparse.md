# Regularization for Sparsity:  
#### L<sub>1</sub> Regularization:
Sparse vectors often contain many dimensions. Creating a feature cross results in even more dimensions. Given such 
high-dimensional feature vectors, model size may become huge and require huge amounts of RAM. A good solution would be 
to reduce many not needed weights to 0. L<sub>2</sub> Regularization encourages the weights to be small but not 0. 
L<sub>1</sub> Regularization encourages many of the uninformative coefficients in the model to be exactly 0 instead of 
a near 0 value.

#### L<sub>1</sub> vs L<sub>2</sub> Regularization:
L<sub>1</sub> and L<sub>2</sub> penalize weights differently:
* L<sub>2</sub> penalizes weight <sup>2</sup>
* L<sub>1</sub> penalizes |weight|

Consequently, L<sub>1</sub> and L<sub>2</sub> have different derivatives:
* The derivative of L<sub>2</sub> is 2 * weight.
* The derivative of L<sub>1</sub> is k (a constant, whose value is independent of weight).

You can think of the derivative of L<sub>2</sub> as a force that removes x% of the weight every time, so L<sub>2</sub>
doesnt drive weights to zero.  

You can think of the derivative of L<sub>1</sub> as a force that subtracts some constant from the weight every time. Due
to the absolute values of L<sub>1</sub> it has a discontinuity at 0.  
**Example:** if a subtraction forces a weight from +0.1 to -0.2 then the actual L<sub>1</sub> will be set to 0.

L<sub>1</sub> Regularization could cause formative features to have 0 weight if:
* Weakly informative features.
* Strongly informative features on different scales.
* Informative features strongly correlated with other similarly informative features.
