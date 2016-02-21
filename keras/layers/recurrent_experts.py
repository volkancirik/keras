# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from keras import activations, initializations
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix, sharedX, shared_ones
from keras.layers.core import Layer, MaskedLayer
from six.moves import range
from keras.layers.recurrent import Recurrent
import math

class ExpertI(Recurrent):
	'''
	Expert Model I
	'''
	def __init__(self, output_dim, n_experts,
				 init='glorot_uniform', inner_init='orthogonal',
				 activation='sigmoid', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, go_backwards=False, **kwargs):
		self.output_dim = output_dim
		self.n_experts = n_experts
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards

		self.input_dim = input_dim
		self.input_length = input_length
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(ExpertI, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_x2e = self.init((self.n_experts,input_dim, self.output_dim))
		self.W_x2g = self.init((input_dim, self.output_dim))

		self.b_x2e = shared_zeros((self.n_experts,self.output_dim))
		self.b_x2g = shared_zeros((self.output_dim))

		self.W_h2e = shared_zeros((self.n_experts, self.output_dim, self.output_dim))

		scale=0.05
		self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, self.n_experts, self.output_dim)))


		self.params = [
			self.W_x2e, self.W_x2g,
			self.b_x2g, self.b_x2e,
			self.W_h2e, self.U_g
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xe_t, xg_t, mask_tm1,
			  h_tm1):
		h_mask_tm1 = mask_tm1 * h_tm1

		E = (xe_t + T.dot(h_mask_tm1, self.W_h2e)).dimshuffle(0,2,1)
		act = T.tensordot( h_mask_tm1 + xg_t, self.U_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		h_t = (gate.reshape((-1,gate.shape[-1])) * E.reshape((-1,E.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])

		return h_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		x_e = T.dot(X, self.W_x2e) + self.b_x2e
		x_g = T.dot(X, self.W_x2g) + self.b_x2g

		outputs, updates = theano.scan(
			self._step,
			sequences=[x_e, x_g, padded_mask],
			outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
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
				  "n_experts": self.n_experts,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length,
				  "go_backwards": self.go_backwards}
		base_config = super(ExpertI, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))




class ExpertII(Recurrent):
	'''
	Expert Model I
	'''
	def __init__(self, output_dim, n_experts,
				 init='glorot_uniform', inner_init='orthogonal',
				 activation='sigmoid', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, go_backwards=False, **kwargs):
		self.output_dim = output_dim
		self.n_experts = n_experts
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards

		self.input_dim = input_dim
		self.input_length = input_length
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(ExpertII, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_x2e = self.init((self.n_experts,input_dim, self.output_dim))
		self.W_e2e = self.init((self.output_dim, self.output_dim))
		self.b_x2e = shared_zeros((self.n_experts,self.output_dim))

		self.W_x2g = self.init((input_dim, self.output_dim))
		self.b_x2g = shared_zeros((self.output_dim))

#		scale=0.05
#		self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, self.n_experts, self.output_dim)))
		self.U_g  = self.init((self.output_dim, self.n_experts, self.output_dim))

		self.params = [
			self.W_x2e, self.W_e2e, self.b_x2e,
			self.W_x2g, self.b_x2g,
			self.U_g
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xe_t, xg_t, mask_tm1,
			  h_tm1, e_tm1):
		h_mask_tm1 = mask_tm1 * h_tm1

		E = (xe_t + T.dot(e_tm1, self.W_e2e)).dimshuffle(0,2,1)
		act = T.tensordot( h_mask_tm1 + xg_t, self.U_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		h_t = (gate.reshape((-1,gate.shape[-1])) * E.reshape((-1,E.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
		e_t = E.dimshuffle(0,2,1)
		return h_t, e_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		x_e = T.dot(X, self.W_x2e) + self.b_x2e
		x_g = T.dot(X, self.W_x2g) + self.b_x2g

		[outputs, expert_memory], updates = theano.scan(
			self._step,
			sequences=[x_e, x_g, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_experts, self.output_dim), 1),
				],
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
				  "n_experts": self.n_experts,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length,
				  "go_backwards": self.go_backwards}
		base_config = super(ExpertII, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class ExpertIIgated(Recurrent):
	'''
	Expert Model I
	'''
	def __init__(self, output_dim, n_experts,
				 init='glorot_uniform', inner_init='orthogonal',
				 activation='sigmoid', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, go_backwards=False, **kwargs):
		self.output_dim = output_dim
		self.n_experts = n_experts
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards

		self.input_dim = input_dim
		self.input_length = input_length
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(ExpertIIgated, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_x2e = self.init((self.n_experts,input_dim, self.output_dim))
		self.W_e2e = self.init((self.output_dim, self.output_dim))
		self.b_x2e = shared_zeros((self.n_experts,self.output_dim))

		self.W_x2g = self.init((input_dim, self.output_dim))
		self.b_x2g = shared_zeros((self.output_dim))

		self.U_g  = self.init((self.output_dim, self.n_experts, self.output_dim))

		self.params = [
			self.W_x2e, self.W_e2e, self.b_x2e,
			self.W_x2g, self.b_x2g,
			self.U_g
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xe_t, xg_t, mask_tm1,
			  h_tm1, e_tm1):
		h_mask_tm1 = mask_tm1 * h_tm1

		e_new = (xe_t + T.dot(e_tm1, self.W_e2e)).dimshuffle(0,2,1)
		act = T.tensordot( h_mask_tm1 + xg_t, self.U_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)
		E = (gate) * e_tm1.dimshuffle(0,2,1) + (1 - gate) * e_new

		h_t = E.sum(axis = 2)
		e_t = E.dimshuffle(0,2,1)
		return h_t, e_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		x_e = T.dot(X, self.W_x2e) + self.b_x2e
		x_g = T.dot(X, self.W_x2g) + self.b_x2g

		[outputs, expert_memory], updates = theano.scan(
			self._step,
			sequences=[x_e, x_g, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.n_experts, self.output_dim), 1),
				],
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
				  "n_experts": self.n_experts,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length,
				  "go_backwards": self.go_backwards}
		base_config = super(ExpertIIgated, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
