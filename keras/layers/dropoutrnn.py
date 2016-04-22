# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations, regularizers
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from ..layers.core import Layer, MaskedLayer
from ..layers.recurrent import Recurrent
from six.moves import range
from ..regularizers import l2

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class DropoutGRU(Recurrent):
	def __init__(self, input_dim, output_dim=128,
				 init='glorot_uniform', inner_init='orthogonal',
				 activation='sigmoid', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 W_regularizer = l2(0.0001), U_regularizer = l2(0.0001), b_regularizer = l2(0.0001),
				 dropout = 0.5):

		super(DropoutGRU, self).__init__()

		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.dropout = dropout

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.input = T.tensor3()

		self.W_z = self.init((self.input_dim, self.output_dim))
		self.U_z = self.inner_init((self.output_dim, self.output_dim))
		self.b_z = shared_zeros((self.output_dim))

		self.W_r = self.init((self.input_dim, self.output_dim))
		self.U_r = self.inner_init((self.output_dim, self.output_dim))
		self.b_r = shared_zeros((self.output_dim))

		self.W_h = self.init((self.input_dim, self.output_dim))
		self.U_h = self.inner_init((self.output_dim, self.output_dim))
		self.b_h = shared_zeros((self.output_dim))

		self.params = [
			self.W_z, self.U_z, self.b_z,
			self.W_r, self.U_r, self.b_r,
			self.W_h, self.U_h, self.b_h,
		]

		self.regularizers = []
		def appendRegulariser(input_regulariser, param, regularizers_list):
			regulariser = regularizers.get(input_regulariser)
			if regulariser:
				regulariser.set_param(param)
				regularizers_list.append(regulariser)

		self.W_regularizer = self.W_regularizer
		appendRegulariser(self.W_regularizer, self.W_z, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_r, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_h, self.regularizers)

		self.U_regularizer = U_regularizer
		appendRegulariser(self.U_regularizer, self.U_z, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_r, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_h, self.regularizers)

		self.b_regularizer = b_regularizer
		appendRegulariser(self.b_regularizer, self.b_z, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_r, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_h, self.regularizers)

		if weights is not None:
			self.set_weights(weights)

	def _step(self,
			  xz_t, xr_t, xh_t, mask_tm1,
			  h_tm1,
			  u_z, u_r, u_h, B_U):
		h_mask_tm1 = mask_tm1 * h_tm1
		z = self.inner_activation(xz_t + T.dot(h_mask_tm1 * B_U[0], u_z))
		r = self.inner_activation(xr_t + T.dot(h_mask_tm1 * B_U[1], u_r))
		hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1 * B_U[2], u_h))
		h_t = z * h_mask_tm1 + (1 - z) * hh_t
		return h_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		retain_prob_W = 1. - self.dropout
		retain_prob_U = 1. - self.dropout
		if train:
			B_W = self.srng.binomial((3, X.shape[1], self.input_dim), 
				p=retain_prob_W, dtype=theano.config.floatX)
			B_U = self.srng.binomial((3, X.shape[1], self.output_dim), 
				p=retain_prob_U, dtype=theano.config.floatX)
		else:
			B_W = np.ones(3, dtype=theano.config.floatX) * retain_prob_W
			B_U = np.ones(3, dtype=theano.config.floatX) * retain_prob_U

		x_z = T.dot(X * B_W[0], self.W_z) + self.b_z
		x_r = T.dot(X * B_W[1], self.W_r) + self.b_r
		x_h = T.dot(X * B_W[2], self.W_h) + self.b_h
		outputs, updates = theano.scan(
			self._step,
			sequences=[x_z, x_r, x_h, padded_mask],
			outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
			non_sequences=[self.U_z, self.U_r, self.U_h, B_U],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		return {"name": self.__class__.__name__,
				"input_dim": self.input_dim,
				"output_dim": self.output_dim,
				"init": self.init.__name__,
				"inner_init": self.inner_init.__name__,
				"activation": self.activation.__name__,
				"inner_activation": self.inner_activation.__name__,
				"truncate_gradient": self.truncate_gradient,
				"return_sequences": self.return_sequences,
				"W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
				"U_regularizer": self.U_regularizer.get_config() if self.b_regularizer else None,
				"b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
#				"W_regularizer": self.W_regularizer,
#				"U_regularizer": self.U_regularizer,
#				"b_regularizer": self.b_regularizer,
				"dropout": self.dropout}


class NaiveDropoutGRU(Recurrent):
	def __init__(self, input_dim, output_dim=128,
				 init='glorot_uniform', inner_init='orthogonal',
				 activation='sigmoid', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 W_regularizer=None, U_regularizer=None, b_regularizer=None, 
				 p_W=0.5, p_U=0.5):

		super(NaiveDropoutGRU, self).__init__()

		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.p_W, self.p_U = shared_zeros((1)), shared_zeros((1))
		self.p_W.set_value(np.array([p_W], dtype=theano.config.floatX))
		self.p_U.set_value(np.array([p_U], dtype=theano.config.floatX))

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.input = T.tensor3()

		self.W_z = self.init((self.input_dim, self.output_dim))
		self.U_z = self.inner_init((self.output_dim, self.output_dim))
		self.b_z = shared_zeros((self.output_dim))

		self.W_r = self.init((self.input_dim, self.output_dim))
		self.U_r = self.inner_init((self.output_dim, self.output_dim))
		self.b_r = shared_zeros((self.output_dim))

		self.W_h = self.init((self.input_dim, self.output_dim))
		self.U_h = self.inner_init((self.output_dim, self.output_dim))
		self.b_h = shared_zeros((self.output_dim))

		self.params = [
			self.W_z, self.U_z, self.b_z,
			self.W_r, self.U_r, self.b_r,
			self.W_h, self.U_h, self.b_h,
		]

		self.regularizers = []
		def appendRegulariser(input_regulariser, param, regularizers_list):
			regulariser = regularizers.get(input_regulariser)
			if regulariser:
				regulariser.set_param(param)
				regularizers_list.append(regulariser)

		self.W_regularizer = W_regularizer
		appendRegulariser(self.W_regularizer, self.W_z, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_r, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_h, self.regularizers)

		self.U_regularizer = U_regularizer
		appendRegulariser(self.U_regularizer, self.U_z, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_r, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_h, self.regularizers)

		self.b_regularizer = b_regularizer
		appendRegulariser(self.b_regularizer, self.b_z, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_r, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_h, self.regularizers)

		if weights is not None:
			self.set_weights(weights)

	def _step(self,
			  xz_t, xr_t, xh_t, mask_tm1, B_U,
			  h_tm1,
			  u_z, u_r, u_h):
		h_mask_tm1 = mask_tm1 * h_tm1
		z = self.inner_activation(xz_t + T.dot(h_mask_tm1 * B_U[0], u_z))
		r = self.inner_activation(xr_t + T.dot(h_mask_tm1 * B_U[1], u_r))
		hh_t = self.activation(xh_t + T.dot(r * h_mask_tm1 * B_U[2], u_h))
		h_t = z * h_mask_tm1 + (1 - z) * hh_t
		return h_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		retain_prob_W = 1. - self.p_W[0]
		retain_prob_U = 1. - self.p_U[0]
		if train:
			B_W = self.srng.binomial((3, X.shape[0], X.shape[1], X.shape[2]), 
				p=retain_prob_W, dtype=theano.config.floatX)
			B_U = self.srng.binomial((X.shape[0], 3, X.shape[1], self.output_dim), 
				p=retain_prob_U, dtype=theano.config.floatX)
		else:
			B_W = np.ones(3, dtype=theano.config.floatX) * retain_prob_W
			B_U = T.ones((X.shape[0], 3), dtype=theano.config.floatX) * retain_prob_U

		x_z = T.dot(X * B_W[0], self.W_z) + self.b_z
		x_r = T.dot(X * B_W[1], self.W_r) + self.b_r
		x_h = T.dot(X * B_W[2], self.W_h) + self.b_h
		outputs, updates = theano.scan(
			self._step,
			sequences=[x_z, x_r, x_h, padded_mask, B_U],
			outputs_info=T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
			non_sequences=[self.U_z, self.U_r, self.U_h],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		return {"name": self.__class__.__name__,
				"input_dim": self.input_dim,
				"output_dim": self.output_dim,
				"init": self.init.__name__,
				"inner_init": self.inner_init.__name__,
				"activation": self.activation.__name__,
				"inner_activation": self.inner_activation.__name__,
				"truncate_gradient": self.truncate_gradient,
				"return_sequences": self.return_sequences,
				"W_regularizer": self.W_regularizer,
				"U_regularizer": self.U_regularizer,
				"b_regularizer": self.b_regularizer,
				"p_W": self.p_W, 
				"p_U": self.p_U}

class DropoutLSTM(Recurrent):
	def __init__(self, input_dim, output_dim=128,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False,
				 W_regularizer = l2(0.0001), U_regularizer = l2(0.0001), b_regularizer = l2(0.0001),
#				 W_regularizer = None, U_regularizer = None, b_regularizer = None,
				 dropout = 0.5):

		super(DropoutLSTM, self).__init__()

		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.dropout = dropout

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.input = T.tensor3()

		self.W_i = self.init((self.input_dim, self.output_dim))
		self.U_i = self.inner_init((self.output_dim, self.output_dim))
		self.b_i = shared_zeros((self.output_dim))

		self.W_f = self.init((self.input_dim, self.output_dim))
		self.U_f = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = self.forget_bias_init((self.output_dim))

		self.W_c = self.init((self.input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((self.input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
		]

		self.regularizers = []
		def appendRegulariser(input_regulariser, param, regularizers_list):
			regulariser = regularizers.get(input_regulariser)
			if regulariser:
				regulariser.set_param(param)
				regularizers_list.append(regulariser)

		self.W_regularizer = W_regularizer
		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_c, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_o, self.regularizers)

		self.U_regularizer = U_regularizer
		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_f, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_c, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_o, self.regularizers)

		self.b_regularizer = b_regularizer
		appendRegulariser(self.b_regularizer, self.b_i, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_f, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_c, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_o, self.regularizers)

		if weights is not None:
			self.set_weights(weights)

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c, B_U):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * B_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * B_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * B_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * B_U[3], u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		retain_prob_W = 1. - self.dropout
		retain_prob_U = 1. - self.dropout
		if train:
			B_W = self.srng.binomial((4, X.shape[1], self.input_dim),
				p=retain_prob_W, dtype=theano.config.floatX)
			B_U = self.srng.binomial((4, X.shape[1], self.output_dim),
				p=retain_prob_U, dtype=theano.config.floatX)
		else:
			B_W = np.ones(4, dtype=theano.config.floatX) * retain_prob_W
			B_U = np.ones(4, dtype=theano.config.floatX) * retain_prob_U

		xi = T.dot(X * B_W[0], self.W_i) + self.b_i
		xf = T.dot(X * B_W[1], self.W_f) + self.b_f
		xc = T.dot(X * B_W[2], self.W_c) + self.b_c
		xo = T.dot(X * B_W[3], self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, B_U],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		return {"name": self.__class__.__name__,
				"input_dim": self.input_dim,
				"output_dim": self.output_dim,
				"init": self.init.__name__,
				"inner_init": self.inner_init.__name__,
				"forget_bias_init": self.forget_bias_init.__name__,
				"activation": self.activation.__name__,
				"inner_activation": self.inner_activation.__name__,
				"truncate_gradient": self.truncate_gradient,
				"return_sequences": self.return_sequences,
				"W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
				"U_regularizer": self.U_regularizer.get_config() if self.b_regularizer else None,
				"b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
#				"W_regularizer": self.W_regularizer,
#				"U_regularizer": self.U_regularizer,
#				"b_regularizer": self.b_regularizer,
				"dropout": self.dropout
				}


class MultiplicativeGaussianNoiseLSTM(Recurrent):
	def __init__(self, input_dim, output_dim=128,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False, 
				 p_W=0.5, p_U=0.5):

		super(MultiplicativeGaussianNoiseLSTM, self).__init__()
		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.p_W, self.p_U = shared_zeros((1)), shared_zeros((1))
		self.p_W.set_value(np.array([p_W], dtype=theano.config.floatX))
		self.p_U.set_value(np.array([p_U], dtype=theano.config.floatX))

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.input = T.tensor3()

		self.W_i = self.init((self.input_dim, self.output_dim))
		self.U_i = self.inner_init((self.output_dim, self.output_dim))
		self.b_i = shared_zeros((self.output_dim))

		self.W_f = self.init((self.input_dim, self.output_dim))
		self.U_f = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = self.forget_bias_init((self.output_dim))

		self.W_c = self.init((self.input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((self.input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
		]

		if weights is not None:
			self.set_weights(weights)

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c, N_U):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * N_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * N_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * N_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * N_U[3], u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))


		std_W = T.sqrt(self.p_W[0] / (1.0 - self.p_W[0]))
		std_U = T.sqrt(self.p_U[0] / (1.0 - self.p_U[0]))
		if train:
			N_W = self.srng.normal((4, X.shape[1], self.input_dim), 
				avg=1.0, std=std_W, dtype=theano.config.floatX)
			N_U = self.srng.normal((4, X.shape[1], self.output_dim), 
				avg=1.0, std=std_U, dtype=theano.config.floatX)
		else:
			N_W = np.ones(4, dtype=theano.config.floatX)
			N_U = np.ones(4, dtype=theano.config.floatX)

		xi = T.dot(X * N_W[0], self.W_i) + self.b_i
		xf = T.dot(X * N_W[1], self.W_f) + self.b_f
		xc = T.dot(X * N_W[2], self.W_c) + self.b_c
		xo = T.dot(X * N_W[3], self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, N_U],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		return {"name": self.__class__.__name__,
				"input_dim": self.input_dim,
				"output_dim": self.output_dim,
				"init": self.init.__name__,
				"inner_init": self.inner_init.__name__,
				"forget_bias_init": self.forget_bias_init.__name__,
				"activation": self.activation.__name__,
				"inner_activation": self.inner_activation.__name__,
				"truncate_gradient": self.truncate_gradient,
				"return_sequences": self.return_sequences,
				"p_W": self.p_W, 
				"p_U": self.p_U}


class NaiveDropoutLSTM(Recurrent):
	def __init__(self, input_dim, output_dim=128,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences=False, 
				 W_regularizer=None, U_regularizer=None, b_regularizer=None, 
				 p_W=0.5, p_U=0.5):

		super(NaiveDropoutLSTM, self).__init__()
		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.truncate_gradient = truncate_gradient
		self.return_sequences = return_sequences
		self.p_W, self.p_U = shared_zeros((1)), shared_zeros((1))
		self.p_W.set_value(np.array([p_W], dtype=theano.config.floatX))
		self.p_U.set_value(np.array([p_U], dtype=theano.config.floatX))

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.input = T.tensor3()

		self.W_i = self.init((self.input_dim, self.output_dim))
		self.U_i = self.inner_init((self.output_dim, self.output_dim))
		self.b_i = shared_zeros((self.output_dim))

		self.W_f = self.init((self.input_dim, self.output_dim))
		self.U_f = self.inner_init((self.output_dim, self.output_dim))
		self.b_f = self.forget_bias_init((self.output_dim))

		self.W_c = self.init((self.input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((self.input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
		]

		self.regularizers = []
		def appendRegulariser(input_regulariser, param, regularizers_list):
			regulariser = regularizers.get(input_regulariser)
			if regulariser:
				regulariser.set_param(param)
				regularizers_list.append(regulariser)

		self.W_regularizer = W_regularizer
		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_o, self.regularizers)

		self.U_regularizer = U_regularizer
		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_f, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_o, self.regularizers)

		self.b_regularizer = b_regularizer
		appendRegulariser(self.b_regularizer, self.b_i, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_f, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_i, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_o, self.regularizers)

		if weights is not None:
			self.set_weights(weights)

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1, B_U,
			  h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * B_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * B_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * B_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * B_U[3], u_o))
		h_t = o_t * self.activation(c_t)
		return h_t, c_t

	def get_output(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		self.train = train

		retain_prob_W = 1. - self.p_W[0]
		retain_prob_U = 1. - self.p_U[0]
		if train:
			B_W = self.srng.binomial((4, X.shape[0], X.shape[1], X.shape[2]), 
				p=retain_prob_W, dtype=theano.config.floatX)
			B_U = self.srng.binomial((X.shape[0], 4, X.shape[1], self.output_dim), 
				p=retain_prob_U, dtype=theano.config.floatX)
		else:
			B_W = T.ones((4), dtype=theano.config.floatX) * retain_prob_W
			B_U = T.ones((X.shape[0], 4), dtype=theano.config.floatX) * retain_prob_U

		xi = T.dot(X * B_W[0], self.W_i) + self.b_i
		xf = T.dot(X * B_W[1], self.W_f) + self.b_f
		xc = T.dot(X * B_W[2], self.W_c) + self.b_c
		xo = T.dot(X * B_W[3], self.W_o) + self.b_o

		[outputs, memories], updates = theano.scan(
			self._step,
			sequences=[xi, xf, xo, xc, padded_mask, B_U],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def get_config(self):
		return {"name": self.__class__.__name__,
				"input_dim": self.input_dim,
				"output_dim": self.output_dim,
				"init": self.init.__name__,
				"inner_init": self.inner_init.__name__,
				"forget_bias_init": self.forget_bias_init.__name__,
				"activation": self.activation.__name__,
				"inner_activation": self.inner_activation.__name__,
				"truncate_gradient": self.truncate_gradient,
				"return_sequences": self.return_sequences,
				"p_W": self.p_W, 
				"p_U": self.p_U}
