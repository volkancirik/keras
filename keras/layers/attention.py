# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from ..layers.recurrent import Recurrent
from ..layers.convolutional import Convolution2D
from ..layers.core import Layer
from six.moves import range

class PreAttention(Layer):
	'''
	pre-process convolution input for attention. it reduces the dimensionality to 3 (n_instances x n_feature_maps x dimension)
	'''
	def __init__(self, **kwargs):
		super(PreAttention, self).__init__(**kwargs)

	def build(self):
		None

	@property
	def output_shape(self):
		return (self.input_shape[0], self.input_shape[1], reduce(lambda x, y: x * y, self.input_shape[2:]))

	def get_output(self, train=False):
		X = self.get_input(train)
		X = X.reshape((X.shape[0],X.shape[1],-1))
		return X

	def get_config(self):
		config = {"name": self.__class__.__name__}
		base_config = super(PreAttention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class DenseAttention(Recurrent):
	'''
	Attention Layer to produce dense representation out of many candidates (n_instances x n_feature_maps or timesteps x dimesion) --> n_instances x dimension
	'''
	def __init__(self,init='glorot_uniform', weights = None, att_dim = None, **kwargs): ###

		self.init = initializations.get(init)
		self.return_sequences = False
		self.initial_weights = weights
		self.att_dim = att_dim
		super(DenseAttention, self).__init__(**kwargs)

	def build(self):
		self.enc_dim = self.input_shape[2]
		self.output_dim = self.enc_dim
		self.input = T.tensor3()

		self.V = self.init((self.att_dim, 1 ))
		self.W_e2a = self.init((self.enc_dim, self.att_dim))


		self.params = [
			self.V, self.W_e2a
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output(self, train = False, get_tuple = False):
		X_encoder = self.get_input(train)
		attention_encoder = T.dot(X_encoder,self.W_e2a)                 # calculate activation
		attention_activation = T.dot( T.tanh(attention_encoder), self.V)         # get score
		attention_alpha = T.nnet.softmax(attention_activation[:,:,0])   # use softmax to calculate weights based on scores
		outputs = (X_encoder * attention_alpha[:,:,None]).sum(axis = 1) # take weighted average

		return outputs

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim": self.output_dim,
				  "att_dim": self.att_dim,
				  "enc_dim": self.enc_dim,
				  "init": self.init.__name__,
				  "return_sequences": self.return_sequences
				  }
		base_config = super(DenseAttention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class TimeDistributedAttention(Recurrent):
	'''
	Attention Layer to produce timedistributed representations out of many candidates  (n_instances x n_feature_maps or timesteps x dimesion) --> n_instances x timesteps x dimension
	'''
	def __init__(self,init='glorot_uniform', weights=None, truncate_gradient=-1, return_sequences=False,
				 go_backwards=False, enc_dim = None, att_dim = None, prev_dim = None, prev_context = True, **kwargs): ###
		self.init = initializations.get(init)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.att_dim = att_dim
		self.prev_dim = prev_dim
		self.enc_dim = enc_dim
		self.prev_context = prev_context

		super(TimeDistributedAttention, self).__init__(**kwargs)

	def build(self):
		self.input = T.tensor3()
		self.enc_dim = self.input_shape[2]
		self.output_dim = self.enc_dim

		self.W_x2a = self.init((self.prev_dim, self.att_dim))     # x_t -> activation
		self.W_e2a = self.init((self.enc_dim, self.att_dim))      # context candidate -> activation
		self.W_ctx2a = self.init((self.enc_dim, self.att_dim))    # previous attention -> activation
		self.V = self.init((self.att_dim, 1 ))                    # activation -> score


		self.params = [
			self.W_x2a, self.V, self.W_e2a
		]
		if self.prev_context:                                     # use previous attention as well
			self.params += [self.W_ctx2a]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,x_t,ctx_tm1,x_encoder, attention_encoder):

		attention_x = T.dot(x_t, self.W_x2a)                           # first part of attention function f x_t -> activation
		attention_total = attention_x[:,None,:] + attention_encoder    # second part x_encoded -> activation
		if self.prev_context:                                          # optional third part previous attended vector -> attention
			attention_prev = T.dot(ctx_tm1,self.W_ctx2a)
			attention_total += attention_prev[:,None,:]

		attention_activation = T.dot( T.tanh(attention_total), self.V)          # attention -> scores
		attention_alpha = T.nnet.softmax(attention_activation[:,:,0])  # scores -> weights
		ctx_t = (x_encoder * attention_alpha[:,:,None]).sum(axis = 1)  # weighted average of context vectors

		return ctx_t

	def get_output(self, train = False, get_tuple = False):

		input_dict = self.get_input(train)
		X_encoder = input_dict['encoder_context']
		X_encoder = X_encoder.reshape((X_encoder.shape[0],X_encoder.shape[1],-1))
		X = input_dict['recurrent_context']
		X = X.dimshuffle((1, 0, 2))

		attention_encoder = T.dot(X_encoder,self.W_e2a)
		outputs, updates = theano.scan(
			self._step,
			sequences=[X],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.enc_dim), 1)
			],
			non_sequences=[X_encoder,attention_encoder],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		if self.return_sequences and self.go_backwards:
			return outputs[::-1].dimshuffle((1, 0, 2))
		elif self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim": self.output_dim,
				  "enc_dim": self.enc_dim,
				  "att_dim": self.att_dim,
				  "prev_dim": self.prev_dim,
				  "prev_context": self.prev_context,
				  "init": self.init.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "go_backwards": self.go_backwards}
		base_config = super(TimeDistributedAttention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class PointerPrediction(Recurrent):
	'''
	Prediction based on Attention
	'''
	def __init__(self,init='glorot_uniform', weights=None, truncate_gradient=-1, return_sequences=False,
				 go_backwards=False, att_dim = None, prev_dim = None, prev_context = False, **kwargs): ###
		self.init = initializations.get(init)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.att_dim = att_dim
		self.prev_dim = prev_dim                                  # use previous attented representation for attention function
		self.prev_context = prev_context

		super(PointerPrediction, self).__init__(**kwargs)

	def build(self):
		self.input = T.tensor3()
		self.enc_dim = self.input_shape[2]

		self.W_x2a = self.init((self.prev_dim, self.att_dim))     # x_t -> activation
		self.W_e2a = self.init((self.enc_dim, self.att_dim))      # context candidate -> activation
		self.W_ctx2a = self.init((self.enc_dim, self.att_dim))    # previous attention -> activation
		self.V = self.init((self.att_dim, 1 ))                    # activation -> score


		self.params = [
			self.W_x2a, self.V, self.W_e2a
		]
		if self.prev_context:                                     # use previous attention as well
			self.params += [self.W_ctx2a]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,x_t,pointer_tm1,ctx_tm1,attentionTotal_tm1,x_encoder, attention_encoder):

		attention_x = T.dot(x_t, self.W_x2a)                           # first part of attention function f x_t -> activation
		attention_total = attention_x[:,None,:] + attention_encoder    # second part x_encoded -> activation
		if self.prev_context:                                          # optional third part previous attended vector -> attention
			attention_prev = T.dot(ctx_tm1,self.W_ctx2a)
			attention_total += attention_prev[:,None,:]

		attention_activation = T.dot( T.tanh(attention_total), self.V)          # attention -> scores
		attention_alpha = T.nnet.softmax( attention_activation[:,:,0])  # scores -> weights
		ctx_t = (x_encoder * attention_alpha[:,:,None]).sum(axis = 1)  # weighted average of context vectors
		pointer_t = attention_alpha
		attentionTotal_t = attention_total
		return pointer_t, ctx_t, attentionTotal_t

	def get_output(self, train = False, get_tuple = False):

		input_dict = self.get_input(train)
		X_encoder = input_dict['encoder_context']
		X_encoder = X_encoder.reshape((X_encoder.shape[0],X_encoder.shape[1],-1))
		X = input_dict['recurrent_context']
		X = X.dimshuffle((1, 0, 2))

		attention_encoder = T.dot(X_encoder,self.W_e2a)
		[outputs, contexts, attentionTotal], updates = theano.scan(
			self._step,
			sequences=[X],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], X.shape[0]), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.enc_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], X.shape[0], self.att_dim), 1)
			],
			non_sequences=[X_encoder,attention_encoder],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		return outputs.dimshuffle((1, 0, 2))


	def debug_output(self, train = False, get_tuple = False):

		input_dict = self.get_input(train)
		X_encoder = input_dict['encoder_context']
		X_encoder = X_encoder.reshape((X_encoder.shape[0],X_encoder.shape[1],-1))
		X = input_dict['recurrent_context']
		X = X.dimshuffle((1, 0, 2))

		attention_encoder = T.dot(X_encoder,self.W_e2a)
		[outputs, contexts, attentionTotal], updates = theano.scan(
			self._step,
			sequences=[X],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], X.shape[0]), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.enc_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], X.shape[0], self.att_dim), 1)
			],
			non_sequences=[X_encoder,attention_encoder],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		return outputs.dimshuffle((1, 0, 2)), X.dimshuffle((1,0,2)), contexts.dimshuffle((1,0,2)), attentionTotal, attention_encoder

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "att_dim": self.att_dim,
				  "prev_dim": self.prev_dim,
				  "prev_context": self.prev_context,
				  "init": self.init.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "go_backwards": self.go_backwards}
		base_config = super(PointerPrediction, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
