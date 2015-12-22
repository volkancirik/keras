# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from keras import activations, initializations
from keras.utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from keras.layers.core import Layer, MaskedLayer
from six.moves import range
from keras.layers.recurrent import Recurrent

class LSTMsum(Recurrent):
	'''
	generalized version of lstms
	'''
	def __init__(self, output_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, **kwargs):
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
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMsum, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_sum = self.init((input_dim, self.output_dim))
		self.U_sum = self.inner_init((self.output_dim, self.output_dim))
		self.b_sum = shared_zeros((self.output_dim))

		self.W_i = self.init((input_dim, self.output_dim))
		self.U_i = self.inner_init((self.output_dim, self.output_dim))
		self.b_i = shared_zeros((self.output_dim))

		self.W_f = self.init((input_dim, self.output_dim))
		self.U_f = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = self.forget_bias_init((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

 		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_sum, self.U_sum, self.b_sum,
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xsum_t, xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  h_tm1, c_tm1,
			  u_sum, u_i, u_f, u_o, u_c):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		sum_t = self.inner_activation(xsum_t + T.dot(h_mask_tm1, u_sum))

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c)) + sum_t * (c_mask_tm1 + self.activation(xc_t + T.dot(h_mask_tm1, u_c)) )
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xsum = T.dot(X, self.W_sum) + self.b_sum
		xi = T.dot(X, self.W_i) + self.b_i
		xf = T.dot(X, self.W_f) + self.b_f
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xsum, xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_sum,self.U_i, self.U_f, self.U_o, self.U_c],
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
		base_config = super(LSTMsum, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class LSTMmul(Recurrent):
	'''
	generalized version of lstms
	'''
	def __init__(self, output_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, **kwargs):
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
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMmul, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_mul = self.init((input_dim, self.output_dim))
		self.U_mul = self.inner_init((self.output_dim, self.output_dim))
		self.b_mul = shared_zeros((self.output_dim))

		self.W_i = self.init((input_dim, self.output_dim))
		self.U_i = self.inner_init((self.output_dim, self.output_dim))
		self.b_i = shared_zeros((self.output_dim))

		self.W_f = self.init((input_dim, self.output_dim))
		self.U_f = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = self.forget_bias_init((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

 		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_mul, self.U_mul, self.b_mul,
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xmul_t, xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  h_tm1, c_tm1,
			  u_mul, u_i, u_f, u_o, u_c):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		mul_t = self.inner_activation(xmul_t + T.dot(h_mask_tm1, u_mul))
		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))

		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c)) + mul_t * (c_mask_tm1 * self.activation(xc_t + T.dot(h_mask_tm1, u_c)) )
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xmul = T.dot(X, self.W_mul) + self.b_mul
		xi = T.dot(X, self.W_i) + self.b_i
		xf = T.dot(X, self.W_f) + self.b_f
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xmul, xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_mul,self.U_i, self.U_f, self.U_o, self.U_c],
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
		base_config = super(LSTMmul, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class LSTMpp(Recurrent):
	'''
	generalized version of lstms
	'''
	def __init__(self, output_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 input_dim=None, input_length=None, **kwargs):
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
		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMpp, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_sum = self.init((input_dim, self.output_dim))               ### add gate weight input weight and bias
		self.U_sum = self.inner_init((self.output_dim, self.output_dim))
		self.b_sum = shared_zeros((self.output_dim))

		self.W_max = self.init((input_dim, self.output_dim))
		self.U_max = self.inner_init((self.output_dim, self.output_dim))
		self.b_max = shared_zeros((self.output_dim))

		self.W_min = self.init((input_dim, self.output_dim))
		self.U_min = self.inner_init((self.output_dim, self.output_dim))
		self.b_min = shared_zeros((self.output_dim))

		self.W_subt = self.init((input_dim, self.output_dim))
		self.U_subt = self.inner_init((self.output_dim, self.output_dim))
		self.b_subt = shared_zeros((self.output_dim))

		self.W_mul = self.init((input_dim, self.output_dim))
		self.U_mul = self.inner_init((self.output_dim, self.output_dim))
		self.b_mul = shared_zeros((self.output_dim))

		self.W_res = self.init((input_dim, self.output_dim))
		self.U_res = self.inner_init((self.output_dim, self.output_dim))
		self.b_res = shared_zeros((self.output_dim))

		self.W_one = self.init((input_dim, self.output_dim))
		self.U_one = self.inner_init((self.output_dim, self.output_dim))
		self.b_one = shared_zeros((self.output_dim))

		self.W_i = self.init((input_dim, self.output_dim))
		self.U_i = self.inner_init((self.output_dim, self.output_dim))
		self.b_i = shared_zeros((self.output_dim))

		self.W_f = self.init((input_dim, self.output_dim))
		self.U_f = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = self.forget_bias_init((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

 		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_sum, self.U_sum, self.b_sum, ### add parameters
			self.W_max, self.U_max, self.b_max,
			self.W_min, self.U_min, self.b_min,
			self.W_subt, self.U_subt, self.b_subt,
			self.W_mul, self.U_mul, self.b_mul,
			self.W_res, self.U_res, self.b_res,
			self.W_one, self.U_one, self.b_one,
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xsum_t, xmax_t, xmin_t, xsubt_t, xmul_t, xres_t, xone_t, xi_t, xf_t, xo_t, xc_t, mask_tm1, ### add op's input x
			  h_tm1, c_tm1,
			  u_sum, u_max, u_min, u_subt, u_mul, u_res, u_one, u_i, u_f, u_o, u_c): ### add gate weight u_ s
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))

		a0_i = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i)) ### gate activations
		a1_f = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
		a2_sum = self.inner_activation(xsum_t + T.dot(h_mask_tm1, u_sum))
		a3_max = self.inner_activation(xmax_t + T.dot(h_mask_tm1, u_max))
		a4_min = self.inner_activation(xmin_t + T.dot(h_mask_tm1, u_min))
		a5_subt = self.inner_activation(xsubt_t + T.dot(h_mask_tm1, u_subt))
		a6_mul = self.inner_activation(xmul_t + T.dot(h_mask_tm1, u_mul))
		a7_res = self.inner_activation(xres_t + T.dot(h_mask_tm1, u_res))
		a8_one = self.inner_activation(xone_t + T.dot(h_mask_tm1, u_one))

		g0_forget = c_mask_tm1
		g1_input = c_tilda
		g2_sum = (c_mask_tm1 + c_tilda)
		g3_max = T.maximum(c_mask_tm1, c_tilda)
		g4_min = T.minimum(c_mask_tm1, c_tilda)
		g5_sub = c_mask_tm1 - c_tilda
		g6_mul = c_mask_tm1 * c_tilda
		g7_res = 0 * c_tilda
		g8_one = 0 * c_tilda + 1

		c_t = a0_i * g0_forget + a1_f * g1_input  + a2_sum * g2_sum + a3_max * g3_max + a4_min * g4_min + a5_subt * g5_sub + a6_mul * g6_mul + a7_res * g7_res + a8_one * g8_one     ### update cell

		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xsum = T.dot(X, self.W_sum) + self.b_sum ### get gate's input
		xmax = T.dot(X, self.W_max) + self.b_max
		xmin = T.dot(X, self.W_min) + self.b_min
		xsubt = T.dot(X, self.W_subt) + self.b_subt
		xmul = T.dot(X, self.W_mul) + self.b_mul
		xres = T.dot(X, self.W_res) + self.b_res
		xone = T.dot(X, self.W_one) + self.b_one

		xi = T.dot(X, self.W_i) + self.b_i
		xf = T.dot(X, self.W_f) + self.b_f
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xsum, xmax, xmin, xsubt, xmul, xres, xone, xi, xf, xo, xc, padded_mask], ### update sequence input
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_sum, self.U_max, self.U_min, self.U_subt, self.U_mul, self.U_res, self.U_one, self.U_i, self.U_f, self.U_o, self.U_c], ### add gate's weight matrix
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
		base_config = super(LSTMpp, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
