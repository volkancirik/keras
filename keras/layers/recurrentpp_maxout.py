# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from keras import activations, initializations
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix, sharedX
from keras.layers.core import Layer, MaskedLayer
from six.moves import range
from keras.layers.recurrent import Recurrent

class LSTM_maxout(Recurrent):
	'''
	soft selection of gates
	'''
	def __init__(self, output_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None,n_pieces = 2, n_opt = 3, **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.input_dim = input_dim
		self.input_length = input_length
		self.n_pieces = n_pieces
		self.n_opt = n_opt
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTM_maxout, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		scale=0.05

		self.W_maxout = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.n_opt, 2 , self.n_pieces)))
		self.b_maxout = shared_zeros((self.output_dim, self.n_opt, self.n_pieces))

		self.W_g = self.init((input_dim, self.output_dim))
		self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, self.n_opt , self.output_dim)))
		self.b_g = shared_zeros((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_maxout, self.b_maxout,
			self.W_g, self.U_g, self.b_g,
			self.W_c, self.U_c, self.b_c,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c, w_maxout, b_maxout):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		ops = [c_mask_tm1,c_tilda]
		y = T.as_tensor_variable( ops, name='y')
		yshuff = T.max(T.dot(y.dimshuffle(1,2,0), w_maxout) + b_maxout, axis = 3)

		c_t = (gate.reshape((-1,gate.shape[-1])) * yshuff.reshape((-1,yshuff.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xg = T.dot(X, self.W_g) + self.b_g
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xg, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_g, self.U_o, self.U_c, self.W_maxout, self.b_maxout],
			truncate_gradient=self.truncate_gradient)



		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim": self.output_dim,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "forget_bias_init": self.forget_bias_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length}
		base_config = super(LSTM_maxout, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class LSTM_maxout_proj(Recurrent):
	'''
	soft selection of gates
	'''
	def __init__(self, output_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None,n_pieces = 2, n_opt = 3, **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.input_dim = input_dim
		self.input_length = input_length
		self.n_pieces = n_pieces
		self.n_opt = n_opt
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTM_maxout_proj, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		scale=0.05

#		self.W_maxout = sharedX(np.random.uniform(low=-scale, high=scale, size=(2, self.n_opt , self.n_pieces)))
		self.W_maxout_1 = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.n_opt , self.output_dim,self.output_dim,self.n_pieces)))
		self.W_maxout_2 = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.n_opt , self.output_dim,self.output_dim,self.n_pieces)))

		self.b_maxout = shared_zeros(((self.n_opt, self.output_dim, self.n_pieces)))

		self.W_g = self.init((input_dim, self.output_dim))
		self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, self.n_opt , self.output_dim)))
		self.b_g = shared_zeros((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
#			self.W_maxout, self.b_maxout,
			self.W_maxout_1, self.W_maxout_2, self.b_maxout,
			self.W_g, self.U_g, self.b_g,
			self.W_c, self.U_c, self.b_c,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

#	def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c, w_maxout, b_maxout):
	def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c, w_maxout_1,w_maxout_2, b_maxout):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
#		ops = [c_mask_tm1,c_tilda]
#		y = T.as_tensor_variable( ops, name='y')
#		yshuff = T.max(T.dot(y.dimshuffle(1,2,0), w_maxout.dimshuffle(1,0,2)) + b_maxout, axis = 3)

		yshuff = T.max(T.dot(c_tilda, w_maxout_1) + T.dot(c_mask_tm1, w_maxout_2) + b_maxout,axis = 3).dimshuffle(0,2,1)

		c_t = (gate.reshape((-1,gate.shape[-1])) * yshuff.reshape((-1,yshuff.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xg = T.dot(X, self.W_g) + self.b_g
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xg, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_g, self.U_o, self.U_c, self.W_maxout_1,self.W_maxout_2, self.b_maxout],
			truncate_gradient=self.truncate_gradient)

#			non_sequences=[self.U_g, self.U_o, self.U_c, self.W_maxout, self.b_maxout],

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim": self.output_dim,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "forget_bias_init": self.forget_bias_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length}
		base_config = super(LSTM_maxout_proj, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class RNN_maxout(Recurrent):
	'''
	'''
	def __init__(self, output_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='tanh',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, go_backwards=False, n_pieces = 2, **kwargs):
		self.output_dim = output_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.n_pieces = n_pieces

		self.input_dim = input_dim
		self.input_length = input_length
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(RNN_maxout, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		scale=0.05
		self.W_maxout = sharedX(np.random.uniform(low=-scale, high=scale, size=(2, self.n_pieces)))
		self.b_maxout = shared_zeros(((self.output_dim, self.n_pieces)))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_maxout, self.b_maxout,
			self.W_c, self.U_c, self.b_c,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self, xo_t, xc_t, mask_tm1, h_tm1, c_tm1, u_o, u_c, w_maxout, b_maxout):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		ops = [c_mask_tm1,c_tilda]
		y = T.as_tensor_variable( ops, name='y')
		c_t = T.max(T.dot(y.dimshuffle(1,2,0), w_maxout) + b_maxout, axis = 2)
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_o, self.U_c, self.W_maxout, self.b_maxout],
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
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "forget_bias_init": self.forget_bias_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "return_sequences": self.return_sequences,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length,
				  "go_backwards": self.go_backwards}
		base_config = super(RNN_maxout, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
