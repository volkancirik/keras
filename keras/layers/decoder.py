# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from ..layers.core import Layer, MaskedLayer
from six.moves import range
from ..layers.recurrent import Recurrent

class LSTMdecoder(Recurrent):
	'''
	sequence decoding using LSTM.
	'''
	def __init__(self, output_dim, hidden_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences = True,
				 enc_name = '', dec_input_name = '',
				 input_dim=None, input_length=None, go_backwards=False, **kwargs):
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.return_sequences = True
		self.input_dim = input_dim
		self.input_length = input_length

		self.enc_name = enc_name
		self.dec_input_name = dec_input_name

		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMdecoder, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_i = self.init((input_dim, self.hidden_dim))
		self.U_i = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_i = shared_zeros((self.hidden_dim))

		self.W_f = self.init((input_dim, self.hidden_dim))
		self.U_f = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_f = self.forget_bias_init((self.hidden_dim))

		self.W_c = self.init((input_dim, self.hidden_dim))
		self.U_c = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_c = shared_zeros((self.hidden_dim))

		self.W_o = self.init((input_dim, self.hidden_dim))
		self.U_o = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_o = shared_zeros((self.hidden_dim))

		self.U_p = self.init((self.hidden_dim, self.output_dim))
		self.b_p = shared_zeros((self.output_dim))

		self.params = [
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
			self.U_p, self.b_p
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred_tm1, h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		pred_t = T.dot(h_t, self.U_p) + self.b_p
		pred_t = T.nnet.softmax(pred_t.reshape((-1, pred_t.shape[-1]))).reshape(pred_t.shape)
		return pred_t, h_t, c_t


	def _step_test(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred_tm1, h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c):

		xi_t = T.dot(pred_tm1, self.W_i) + self.b_i
		xf_t = T.dot(pred_tm1, self.W_f) + self.b_f
		xc_t = T.dot(pred_tm1, self.W_c) + self.b_c
		xo_t = T.dot(pred_tm1, self.W_o) + self.b_o

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		pred_t = T.dot(h_t, self.U_p) + self.b_p
		pred_t = T.nnet.softmax(pred_t.reshape((-1, pred_t.shape[-1]))).reshape(pred_t.shape)
		pred_t = T.ge(pred_t, T.max(pred_t, axis = 1).reshape((pred_t.shape[0],1)))*1.0
		return pred_t, h_t, c_t

	def get_output(self, train=False):

		input_dict = self.get_input(train)
		X = input_dict[self.dec_input_name]
		prev_state = input_dict[self.enc_name]

		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xi = T.dot(X, self.W_i) + self.b_i
		xf = T.dot(X, self.W_f) + self.b_f
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		if train:
			STEP = self._step
		else:
			STEP = self._step_test
		[outputs, hiddens, memories], updates = theano.scan(
			STEP,
			sequences=[xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(prev_state, 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		if self.go_backwards:
			return outputs[::-1].dimshuffle((1, 0, 2))
		return outputs.dimshuffle((1, 0, 2))

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim": self.output_dim,
				  "hidden_dim": self.hidden_dim,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "forget_bias_init": self.forget_bias_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length,
				  "return_sequences": self.return_sequences,
				  "go_backwards": self.go_backwards,
				  "enc_name" : self.enc_name,
				  "dec_input_name" : self.dec_input_name
		}
		base_config = super(LSTMdecoder, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))




class LSTMhdecoder(Recurrent):
	'''
	sequence decoding using LSTM.
	'''
	def __init__(self, output_dim, hidden_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences = True,
				 enc_name = '', dec_input_name = '',
				 input_dim=None, input_length=None, go_backwards=False, **kwargs):
		self.output_dim = output_dim
		self.hidden_dim = hidden_dim
		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.return_sequences = True
		self.input_dim = input_dim
		self.input_length = input_length

		self.enc_name = enc_name
		self.dec_input_name = dec_input_name

		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMhdecoder, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		self.W_i = self.init((input_dim, self.hidden_dim))
		self.U_i = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_i = shared_zeros((self.hidden_dim))

		self.W_f = self.init((input_dim, self.hidden_dim))
		self.U_f = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_f = self.forget_bias_init((self.hidden_dim))

		self.W_c = self.init((input_dim, self.hidden_dim))
		self.U_c = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_c = shared_zeros((self.hidden_dim))

		self.W_o = self.init((input_dim, self.hidden_dim))
		self.U_o = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_o = shared_zeros((self.hidden_dim))

		self.U_p1 = self.init((self.hidden_dim, self.output_dim))
		self.b_p1 = shared_zeros((self.output_dim))

		self.U_p2 = self.init((self.hidden_dim, self.output_dim))
		self.b_p2 = shared_zeros((self.output_dim))

		self.U_p3 = self.init((self.hidden_dim, self.output_dim))
		self.b_p3 = shared_zeros((self.output_dim))

		self.U_p4 = self.init((self.hidden_dim, self.output_dim))
		self.b_p4 = shared_zeros((self.output_dim))


		self.params = [
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
			self.U_p1, self.b_p1,
			self.U_p2, self.b_p2,
			self.U_p3, self.b_p3,
			self.U_p4, self.b_p4,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred1_tm1, pred2_tm1, pred3_tm1, pred4_tm1, h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)

		pred1_t = T.dot(h_t, self.U_p1) + self.b_p1
		pred1_t = T.nnet.softmax(pred1_t.reshape((-1, pred1_t.shape[-1]))).reshape(pred1_t.shape)

		pred2_t = T.dot(h_t, self.U_p2) + self.b_p2
		pred2_t = T.nnet.softmax(pred2_t.reshape((-1, pred2_t.shape[-1]))).reshape(pred2_t.shape)

		pred3_t = T.dot(h_t, self.U_p3) + self.b_p3
		pred3_t = T.nnet.softmax(pred3_t.reshape((-1, pred3_t.shape[-1]))).reshape(pred3_t.shape)

		pred4_t = T.dot(h_t, self.U_p4) + self.b_p4
		pred4_t = T.nnet.softmax(pred4_t.reshape((-1, pred4_t.shape[-1]))).reshape(pred4_t.shape)

		return pred1_t, pred2_t, pred3_t, pred4_t, h_t, c_t

	def _step_test(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred1_tm1, pred2_tm1, pred3_tm1, pred4_tm1, h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c):

		outer1 = pred1_tm1[:, :, np.newaxis] * pred2_tm1[:, np.newaxis, :]
		outer1 =  outer1.reshape((outer1.shape[0],-1))

		outer2 = pred3_tm1[:, :, np.newaxis] * pred4_tm1[:, np.newaxis, :]
		outer2 =  outer2.reshape((outer2.shape[0],-1))

		pred = outer1[:, :, np.newaxis] * outer2[:, np.newaxis, :]
		pred =  pred.reshape((pred.shape[0],-1))

		xi_t = T.dot(pred, self.W_i) + self.b_i
		xf_t = T.dot(pred, self.W_f) + self.b_f
		xc_t = T.dot(pred, self.W_c) + self.b_c
		xo_t = T.dot(pred, self.W_o) + self.b_o

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1, u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1, u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)

		pred1_t = T.dot(h_t, self.U_p1) + self.b_p1
		pred1_t = T.nnet.softmax(pred1_t.reshape((-1, pred1_t.shape[-1]))).reshape(pred1_t.shape)

		pred2_t = T.dot(h_t, self.U_p2) + self.b_p2
		pred2_t = T.nnet.softmax(pred2_t.reshape((-1, pred2_t.shape[-1]))).reshape(pred2_t.shape)

		pred3_t = T.dot(h_t, self.U_p3) + self.b_p3
		pred3_t = T.nnet.softmax(pred3_t.reshape((-1, pred3_t.shape[-1]))).reshape(pred3_t.shape)

		pred4_t = T.dot(h_t, self.U_p4) + self.b_p4
		pred4_t = T.nnet.softmax(pred4_t.reshape((-1, pred4_t.shape[-1]))).reshape(pred4_t.shape)

		pred1_t = T.ge(pred1_t, T.max(pred1_t, axis = 1).reshape((pred1_t.shape[0],1)))*1.0
		pred2_t = T.ge(pred2_t, T.max(pred2_t, axis = 1).reshape((pred2_t.shape[0],1)))*1.0
		pred3_t = T.ge(pred3_t, T.max(pred3_t, axis = 1).reshape((pred3_t.shape[0],1)))*1.0
		pred4_t = T.ge(pred4_t, T.max(pred4_t, axis = 1).reshape((pred4_t.shape[0],1)))*1.0

		return pred1_t, pred2_t, pred3_t, pred4_t, h_t, c_t

	def get_output(self, train=False):

		input_dict = self.get_input(train)
		X = input_dict[self.dec_input_name]
		prev_state = input_dict[self.enc_name]

		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xi = T.dot(X, self.W_i) + self.b_i
		xf = T.dot(X, self.W_f) + self.b_f
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		if train:
			STEP = self._step
		else:
			STEP = self._step_test
		[outputs1, outputs2, outputs3, outputs4, hiddens, memories], updates = theano.scan(
			STEP,
			sequences=[xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(prev_state, 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		if self.go_backwards:
			outputs1 = outputs1[::-1]
			outputs2 = outputs2[::-1]
			outputs3 = outputs3[::-1]
			outputs4 = outputs4[::-1]

		return [outputs1.dimshuffle((1, 0, 2)), outputs2.dimshuffle((1, 0, 2)), outputs3.dimshuffle((1, 0, 2)), outputs4.dimshuffle((1, 0, 2))]

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim": self.output_dim,
				  "hidden_dim": self.hidden_dim,
				  "init": self.init.__name__,
				  "inner_init": self.inner_init.__name__,
				  "forget_bias_init": self.forget_bias_init.__name__,
				  "activation": self.activation.__name__,
				  "inner_activation": self.inner_activation.__name__,
				  "truncate_gradient": self.truncate_gradient,
				  "input_dim": self.input_dim,
				  "input_length": self.input_length,
				  "return_sequences": self.return_sequences,
				  "go_backwards": self.go_backwards,
				  "enc_name" : self.enc_name,
				  "dec_input_name" : self.dec_input_name
		}
		base_config = super(LSTMhdecoder, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class SplitDecoder(Layer):
	'''
	'''
	def __init__(self, index, **kwargs):
		super(SplitDecoder, self).__init__(**kwargs)
		self.index = index
		self.input = T.tensor4()

	def get_output(self, train=False):
		X = self.get_input(train)
		return X[self.index]

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "index": self.index}
		base_config = super(SplitDecoder, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
