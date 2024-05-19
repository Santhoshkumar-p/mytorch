import sys
sys.path.append('mytorch')

from Conv2d import *
from activation import *
from batchnorm2d import *

import numpy as np
import os


class ConvBlock(object):
	def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
		#TODO	
		self.layers = [
			Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            BatchNorm2d(out_channels)
		] 											

	def forward(self, A):
		for l in self.layers:
			A = l.forward(A)
		return A

	def backward(self, grad): 
		for l in reversed(self.layers):
			grad = l.backward(grad)
		return grad

class ResBlock(object):
	def __init__(self, in_channels, out_channels, filter_size, stride=3, padding=1):
		self.convolution_layers =  [
			ConvBlock(in_channels, out_channels, filter_size, stride, padding),
			ReLU(),
			ConvBlock(out_channels, out_channels,  1, 1, 0) #1,1,0
		 ] #TODO Initialize all layers in this list.				
		self.final_activation =	ReLU()				#TODO 

		if stride != 1 or in_channels != out_channels or filter_size!=1 or padding!=0:
			self.residual_connection = ConvBlock(in_channels, out_channels, filter_size, stride, padding) 		#TODO
		else:
			self.residual_connection = None			#TODO 


	def forward(self, A):
		Z = A
		'''
		Implement the forward for convolution layer.

		'''
		for conv_layer in self.convolution_layers:
			Z = conv_layer.forward(Z)
	
		'''
		Add the residual connection to the output of the convolution layers

		'''
		if self.residual_connection is not None:
			residual = self.residual_connection.forward(A)
			Z = Z + residual
		

		'''
		Pass the the sum of the residual layer and convolution layer to the final activation function
		'''	
		Z = self.final_activation.forward(Z)
		return Z


	def backward(self, grad):

		'''
		Implement the backward of the final activation
		'''
		#TODO 
		grad = self.final_activation.backward(grad)

		'''
		Implement the backward of residual layer to get "residual_grad"
		'''
		#TODO 
		residual_grad = None
		if self.residual_connection is not None:
			residual_grad = self.residual_connection.backward(grad)
			

		'''
		Implement the backward of the convolution layer to get "convlayers_grad"
		'''
		convlayers_grad = grad
		for conv_layer in reversed(self.convolution_layers):
			convlayers_grad = conv_layer.backward(convlayers_grad)

		'''
		Add convlayers_grad and residual_grad to get the final gradient 
		'''

		if self.residual_connection is not None:
			grad = convlayers_grad + residual_grad

		return grad