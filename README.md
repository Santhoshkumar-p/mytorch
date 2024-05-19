# MyTorch Deep Learning Library

## Overview
A custom deep learning library implemented from scratch. MyTorch is inspired by PyTorch and is designed to create various neural network architectures, including multilayer perceptrons (MLP), convolutional neural networks (CNN), and recurrent neural networks (RNN) with gated recurrent units (GRU) and long short-term memory (LSTM) structures.

## File Structure
- **mytorch**: Core directory containing the implementation of MyTorch library.
  - **nn**: Subdirectory containing neural network modules.
    - **linear.py**: Implementation of linear layers with Autograd.
    - **activation.py**: Implementation of activation functions.
    - **loss.py**: Implementation of loss functions.
    - **batchnorm.py**: Implementation of batch normalization.
    - **batchnorm2d.py**: Implementation of batch normalization(https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
    - **dropout.py**: Implementation of Dropout Layer.
    - **dropout2d.py**: Implementation of 2D Dropout Layer.
    - **flatten.py**: Implementation of Flatten Layer(Flattens a contiguous range of dims into a tensor).
    - **conv.py**: Implementation of Convolution Layers with Autograd.
    - **Conv1d.py**: Implementation of 1D Convolution Layers.
    - **Conv2d.py**: Implementation of 2D Convolution Layers.
    - **ConvTranspose.py**: Implementation of 1D & 2D Transposed Convolution Layerys(input values A are upsampled and then convolved). 
    - **pool.py**: Implementation of Mean Pooling and Max Pooling.
    - **resampling_autograd.py**: Implementation of 1D & 2D UpSampling and DownSampling with Autograd details.
    - **resampling_autograd.py**: Implementation of 1D & 2D UpSampling and DownSampling layers.
    - **gru_cell.py**: Implementation of a gated recurrent unit (GRU) cell(https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html).
    - **rnn_cell.py**: Implementation of Vanilla RNN cell.
    - **attention.py**: Implementing a self-attention with a causal additive mask (i.e., masked self-attention).
  - **optim**: Subdirectory containing optimization algorithms.
    - **sgd.py**: Implementation of stochastic gradient descent optimizer.
    - **adam.py**: Implementation of Adam optimizer(Momentum + RMSProp and Adaptive Learning Rate per parameter).
    - **adamW.py**: Implementation of AdamW optimizer(Adam +  Weight decay directly into the update step).
  - **autograd_engine.py**: Implementation of Autograd Engine.
  - **functional_1.py**: Implementation of Autograd Ops for primitive levels.
  - **functional_2.py**: Implementation of Autograd Ops for Convolution Layers.
  - **utils.py**: Implementation of Gradient Buffer for Autograd Engine to keep track of gradients for back propogation.
  - **models**: Subdirectory containing predefined neural network models, mainly to test mytorch implementations.
    - **mlp.py**: Implementation of multilayer perceptrons using mytorch.
    - **mlp_examples.py**: Implementation of multilayer perceptrons using mytorch with different layers and activations.
    - **mlp_scan.py**: Implementation of multilayer perceptrons and CNN scans using mytorch.
    - **resnet.py**: Implementation of ResNet using mytorch.
    - **rnn_classifier.py**: Implementation of RNN Phoneme classifier using mytorch.
    - **char_predictor.py**: Implementation of Simple Character predictor using GRU using mytorch.
  - **CTC**: Subdirectory containing predefined neural network models, mainly to test mytorch implementations.
    - **CTC.py**: Implementation of CTC Loss to train a seq-to-seq model in mytorch
    - **CTCDecoding.py**: Implementation of CTC Decoding: Greedy Search and Beam Search, to decode the model output probabilities
- **requirements.txt**: File specifying required Python packages.

## Details
The library will cover various components such as forward propagation, loss calculation, backward propagation, and gradient descent. 
The first part of the assignment focuses on creating the core components of multilayer perceptrons, including linear layers, activation functions, batch normalization, loss functions, and stochastic gradient descent optimizer.

# Autograd

## Motivation
Neural networks are complex functions, and training them involves calculating derivatives (gradients) with respect to their inputs, which is essential for optimization algorithms like gradient descent. Automating this calculation can be done through symbolic differentiation, numerical differentiation, or automatic differentiation (autodiff). Autodiff, specifically reverse mode autodiff, efficiently computes derivatives of complex functions by repeatedly applying the chain rule.

## Implementation Details
Autograd keeps track of primitive operations performed on input data, enabling backpropagation to calculate gradients efficiently.  
It categorizes into two types:
- Forward accumulation (forward mode)
- Reverse accumulation (reverse mode)   
<br>Note: Autograd employs reverse mode autodiff, which calculates gradients from outside to inside of the chain rule.

### Operation Class
Represents primitive operations in the network. Stores inputs, outputs, gradients to update, and backward function for each operation.

### Autograd Class
Main class responsible for tracking operations sequence and initiating backpropagation.

### GradientBuffer Class
Wrapper class around a dictionary for storing and updating gradients.

## Example Walkthrough
Building a single-layer MLP involves adding nodes for multiplication and addition operations to the computation graph, then invoking backpropagation to update gradients.  
Suppose we're constructing a single-layer MLP, an affine combination expressed as \( y = x ∗ W + b \). This can be broken down into two operations on \( x \):
1. h = x ∗ W    (1)
2. y = h + b    (2)  
To implement this function using our autograd engine, we start by creating an instance of the autograd engine.  

```python
autograd = autograd_engine.Autograd()
```

For equation (1), we need to add a node to the computation graph performing multiplication, which would be done in the following way:

```python
autograd_engine.add_operation(inputs=[x, W], output=h,
                              gradients_to_update=[None, dW],
                              backward_operation=matmul_backward)
```

Similarly for equation (2),  

```python
autograd_engine.add_operation(inputs = [h, b], output = y,
                                gradients_to_update = [None, db],
                                backward_operation = add_backward)
```

Invoke backpropagation by:  

```python
autograd_engine.backward(divergence)
```

dW and db should be updated after this 

## Gradients Handling
Autograd tracks gradients for both input data and network parameters differently. Input data gradients are stored internally, while network parameter gradients are tracked externally.

# Reference
The project developed was completed in the course https://deeplearning.cs.cmu.edu/S24/index.html 
