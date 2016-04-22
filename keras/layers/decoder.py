# -*- coding: utf-8 -*-
from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np

from .. import activations, initializations, regularizers, regularizers
from ..utils.theano_utils import shared_scalar, shared_zeros, alloc_zeros_matrix
from ..layers.core import Layer, MaskedLayer
from six.moves import range
from ..layers.recurrent import Recurrent
from ..regularizers import l2

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class SequenceLayerMerge(Recurrent):
	'''
	merge a sequence layer and a ff layer
	'''
	def __init__(self, output_dim, return_sequences = True,
				 ff_name = '', seq_name = '', **kwargs):

		self.params = []
		self.output_dim = output_dim
		self.return_sequences = True

		self.seq_name = seq_name
		self.ff_name = ff_name

	def _step(self,x_t,o_tm1,ff_layer):
		o_t = T.cast(T.concatenate([x_t,ff_layer],axis = -1),theano.config.floatX)
		return o_t

	def get_output(self, train=False):

		input_dict = self.get_input(train)
		X = input_dict[self.seq_name]
		ff_layer = input_dict[self.ff_name]

		X = X.dimshuffle((1, 0, 2))

		return outputs.dimshuffle((1, 0, 2))

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "output_dim" : self.output_dim,
				  "return_sequences": self.return_sequences,
				  "ff_name" : self.ff_name,
				  "seq_name" : self.seq_name
		}
		base_config = super(SequenceLayerMerge, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

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

		self.We_i = self.init((self.hidden_dim, self.hidden_dim))
		self.We_f = self.init((self.hidden_dim, self.hidden_dim))
		self.We_c = self.init((self.hidden_dim, self.hidden_dim))
		self.We_o = self.init((self.hidden_dim, self.hidden_dim))

		self.params = [
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
			self.U_p, self.b_p,
			self.We_i,self.We_c,self.We_f,self.We_o
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred_tm1, h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c, prev_state):
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
			  u_i, u_f, u_o, u_c, prev_state):

		xi_t = T.dot(pred_tm1, self.W_i) + self.b_i + T.dot(prev_state, self.We_i)
		xf_t = T.dot(pred_tm1, self.W_f) + self.b_f + T.dot(prev_state, self.We_f)
		xc_t = T.dot(pred_tm1, self.W_c) + self.b_c + T.dot(prev_state, self.We_c)
		xo_t = T.dot(pred_tm1, self.W_o) + self.b_o + T.dot(prev_state, self.We_o)

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

		xi = T.dot(X, self.W_i) + self.b_i + T.dot(prev_state, self.We_i)
		xf = T.dot(X, self.W_f) + self.b_f + T.dot(prev_state, self.We_f)
		xc = T.dot(X, self.W_c) + self.b_c + T.dot(prev_state, self.We_c)
		xo = T.dot(X, self.W_o) + self.b_o + T.dot(prev_state, self.We_o)

		if train:
			STEP = self._step
		else:
			STEP = self._step_test
		[outputs, hiddens, memories], updates = theano.scan(
			STEP,
			sequences=[xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, prev_state],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		if self.go_backwards:
			return outputs[::-1].dimshuffle((1, 0, 2))
		return outputs.dimshuffle((1, 0, 2))

	def get_debug(self, train=False):

		input_dict = self.get_input(train)
		X = input_dict[self.dec_input_name]
		prev_state = input_dict[self.enc_name]

		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xi = T.dot(X, self.W_i) + self.b_i + T.dot(prev_state, self.We_i)
		xf = T.dot(X, self.W_f) + self.b_f + T.dot(prev_state, self.We_f)
		xc = T.dot(X, self.W_c) + self.b_c + T.dot(prev_state, self.We_c)
		xo = T.dot(X, self.W_o) + self.b_o + T.dot(prev_state, self.We_o)

		if train:
			STEP = self._step
		else:
			STEP = self._step_test
		[outputs, hiddens, memories], updates = theano.scan(
			STEP,
			sequences=[xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, prev_state],
			truncate_gradient=self.truncate_gradient,
			go_backwards=self.go_backwards)

		return outputs, hiddens, memories, prev_state


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
	def __init__(self, output_dim, hidden_dim, v_dim, input_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences = True,
				 enc_name = '', dec_input_name = '',
				 W_regularizer = l2(0.0001), U_regularizer = l2(0.0001), b_regularizer = l2(0.0001),
				 dropout = 0.5,
				 input_length=None, go_backwards=False, **kwargs):

		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.dropout = dropout
		self.regularizers = []
		self.W_regularizer = W_regularizer
		self.U_regularizer = U_regularizer
		self.b_regularizer = b_regularizer

		self.v_dim = v_dim
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.return_sequences = True
		self.input_length = input_length

		self.enc_name = enc_name
		self.dec_input_name = dec_input_name

		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMhdecoder, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_dim

		self.W_embedding = self.init((self.v_dim, self.input_dim))

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

		self.We_i = self.init((self.hidden_dim, self.hidden_dim))
		self.We_f = self.init((self.hidden_dim, self.hidden_dim))
		self.We_c = self.init((self.hidden_dim, self.hidden_dim))
		self.We_o = self.init((self.hidden_dim, self.hidden_dim))

		self.params = [
			self.W_embedding,
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
			self.U_p1, self.b_p1,
			self.U_p2, self.b_p2,
			self.U_p3, self.b_p3,
			self.U_p4, self.b_p4,
			self.We_i,self.We_c,self.We_f,self.We_o
		]

		def appendRegulariser(input_regulariser, param, regularizers_list):
			regulariser = regularizers.get(input_regulariser)
			if regulariser:
				regulariser.set_param(param)
				regularizers_list.append(regulariser)
		appendRegulariser(self.W_regularizer, self.W_embedding, self.regularizers)
		appendRegulariser(self.W_regularizer, self.We_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.We_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.We_c, self.regularizers)
		appendRegulariser(self.W_regularizer, self.We_o, self.regularizers)

		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_o, self.regularizers)


		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_f, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_o, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p1, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p2, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p3, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p4, self.regularizers)

		appendRegulariser(self.b_regularizer, self.b_i, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_f, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_c, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_o, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p1, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p2, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p3, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p4, self.regularizers)


		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,
			  xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred1_tm1, pred2_tm1, pred3_tm1, pred4_tm1, h_tm1, c_tm1,
			  u_i, u_f, u_o, u_c, prev_state, B_W, B_U, B_We):
		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * B_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * B_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * B_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * B_U[3], u_o))
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
				   u_i, u_f, u_o, u_c, prev_state, B_W, B_U, B_We):

		outer1 = pred1_tm1[:, :, np.newaxis] * pred2_tm1[:, np.newaxis, :]
		outer1 =  outer1.reshape((outer1.shape[0],-1))

		outer2 = pred3_tm1[:, :, np.newaxis] * pred4_tm1[:, np.newaxis, :]
		outer2 =  outer2.reshape((outer2.shape[0],-1))

		pred = outer1[:, :, np.newaxis] * outer2[:, np.newaxis, :]
		pred =	pred.reshape((pred.shape[0],-1))

		xi_t = T.dot(self.W_embedding[T.argmax(pred, axis = 1)] * B_W[0], self.W_i) + self.b_i + T.dot(prev_state * B_We[0], self.We_i)
		xf_t = T.dot(self.W_embedding[T.argmax(pred, axis = 1)] * B_W[1], self.W_f) + self.b_f + T.dot(prev_state * B_We[1], self.We_f)
		xc_t = T.dot(self.W_embedding[T.argmax(pred, axis = 1)] * B_W[2], self.W_c) + self.b_c + T.dot(prev_state * B_We[2], self.We_c)
		xo_t = T.dot(self.W_embedding[T.argmax(pred, axis = 1)] * B_W[3], self.W_o) + self.b_o + T.dot(prev_state * B_We[3], self.We_o)

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * B_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * B_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * B_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * B_U[3], u_o))
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
		X_idx = input_dict[self.dec_input_name]
		X = self.W_embedding[X_idx]

		prev_state = input_dict[self.enc_name]
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		retain_prob = 1. - self.dropout

		if train:
			B_W = self.srng.binomial((4, X.shape[1], self.input_dim),  p=retain_prob, dtype=theano.config.floatX)
			B_U = self.srng.binomial((4, X.shape[1], self.hidden_dim), p=retain_prob, dtype=theano.config.floatX)
			B_We = self.srng.binomial((4, X.shape[1], self.hidden_dim),	 p=retain_prob, dtype=theano.config.floatX)
		else:
			B_W = np.ones(4, dtype=theano.config.floatX) * retain_prob
			B_U = np.ones(4, dtype=theano.config.floatX) * retain_prob
			B_We = np.ones(4, dtype=theano.config.floatX) * retain_prob

		xi = T.dot(X * B_W[0], self.W_i) + self.b_i + T.dot(prev_state * B_We[0], self.We_i)
		xf = T.dot(X * B_W[1], self.W_f) + self.b_f + T.dot(prev_state * B_We[1], self.We_f)
		xc = T.dot(X * B_W[2], self.W_c) + self.b_c + T.dot(prev_state * B_We[2], self.We_c)
		xo = T.dot(X * B_W[3], self.W_o) + self.b_o + T.dot(prev_state * B_We[3], self.We_o)

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
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim), 1)
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, prev_state, B_W, B_U, B_We],
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
				  "v_dim": self.v_dim,
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
				  "dec_input_name" : self.dec_input_name,
				  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
				  "U_regularizer": self.U_regularizer.get_config() if self.b_regularizer else None,
				  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
				  "dropout" : self.dropout
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

####
class LSTMAttentionDecoder(Recurrent):
	'''
	Attention decoder
	'''
	def __init__(self, output_dim, hidden_dim, v_dim, input_dim,
				 init='glorot_uniform', inner_init='orthogonal', forget_bias_init='one',
				 activation='tanh', inner_activation='hard_sigmoid',
				 weights=None, truncate_gradient=-1, return_sequences = True,
				 enc_name = '', dec_input_name = '', img_name = '',prev_context = True,
				 W_regularizer = l2(0.0001), U_regularizer = l2(0.0001), b_regularizer = l2(0.0001),
				 dropout = 0.5,
				 input_length=None, go_backwards=False, **kwargs):

		self.srng = RandomStreams(seed=np.random.randint(10e6))
		self.dropout = dropout
		self.regularizers = []
		self.W_regularizer = W_regularizer
		self.U_regularizer = U_regularizer
		self.b_regularizer = b_regularizer

		self.v_dim = v_dim
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		self.init = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.activation = activations.get(activation)
		self.inner_activation = activations.get(inner_activation)
		self.truncate_gradient = truncate_gradient
		self.initial_weights = weights
		self.go_backwards = go_backwards
		self.return_sequences = True
		self.input_length = input_length

		self.enc_name = enc_name
		self.dec_input_name = dec_input_name
		self.img_name = img_name

		self.prev_context = prev_context

		if self.input_dim:
			kwargs['input_shape'] = (self.input_length, self.input_dim)
		super(LSTMAttentionDecoder, self).__init__(**kwargs)

	def build(self):

		self.W_embedding = self.init((self.v_dim, self.input_dim))

		self.W_x2a = self.init((self.input_dim, self.hidden_dim))			# x_t -> activation
		self.W_e2a = self.init((self.hidden_dim, self.hidden_dim))		# context candidate -> activation
		self.W_ctx2a = self.init((self.hidden_dim, self.hidden_dim))	# previous attention -> activation
		self.V = self.init((self.hidden_dim, 1 ))						# activation -> score

		self.W_i = self.init((self.input_dim, self.hidden_dim))
		self.U_i = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_i = shared_zeros((self.hidden_dim))

		self.W_f = self.init((self.input_dim, self.hidden_dim))
		self.U_f = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_f = self.forget_bias_init((self.hidden_dim))

		self.W_c = self.init((self.input_dim, self.hidden_dim))
		self.U_c = self.inner_init((self.hidden_dim, self.hidden_dim))
		self.b_c = shared_zeros((self.hidden_dim))

		self.W_o = self.init((self.input_dim, self.hidden_dim))
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

		self.Wimg_i = self.init((self.hidden_dim, self.hidden_dim))
		self.Wimg_f = self.init((self.hidden_dim, self.hidden_dim))
		self.Wimg_c = self.init((self.hidden_dim, self.hidden_dim))
		self.Wimg_o = self.init((self.hidden_dim, self.hidden_dim))

		self.Wctx_i = self.init((self.hidden_dim, self.hidden_dim))
		self.Wctx_f = self.init((self.hidden_dim, self.hidden_dim))
		self.Wctx_c = self.init((self.hidden_dim, self.hidden_dim))
		self.Wctx_o = self.init((self.hidden_dim, self.hidden_dim))

		self.params = [
			self.W_embedding,
			self.Wimg_i,
			self.Wimg_f,
			self.Wimg_c,
			self.Wimg_o,
			self.W_i, self.U_i, self.b_i,
			self.W_c, self.U_c, self.b_c,
			self.W_f, self.U_f, self.b_f,
			self.W_o, self.U_o, self.b_o,
			self.U_p1, self.b_p1,
			self.U_p2, self.b_p2,
			self.U_p3, self.b_p3,
			self.U_p4, self.b_p4,
			self.V, self.W_e2a, self.W_x2a,
			self.Wctx_i,self.Wctx_f,self.Wctx_c,self.Wctx_o
		]
		def appendRegulariser(input_regulariser, param, regularizers_list):
			regulariser = regularizers.get(input_regulariser)
			if regulariser:
				regulariser.set_param(param)
				regularizers_list.append(regulariser)

		if self.prev_context:									  # use previous attention as well
			self.params += [self.W_ctx2a]
			appendRegulariser(self.W_regularizer, self.W_ctx2a, self.regularizers)

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

		appendRegulariser(self.W_regularizer, self.W_embedding, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_x2a, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_e2a, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_ctx2a, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.W_o, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wimg_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wimg_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wimg_c, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wimg_o, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wctx_i, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wctx_f, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wctx_c, self.regularizers)
		appendRegulariser(self.W_regularizer, self.Wctx_o, self.regularizers)

		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_f, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_i, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_o, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p1, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p2, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p3, self.regularizers)
		appendRegulariser(self.U_regularizer, self.U_p4, self.regularizers)

		appendRegulariser(self.b_regularizer, self.b_i, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_f, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_c, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_o, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p1, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p2, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p3, self.regularizers)
		appendRegulariser(self.b_regularizer, self.b_p4, self.regularizers)

	def _step(self,
			  x_t, xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred1_tm1, pred2_tm1, pred3_tm1, pred4_tm1, h_tm1, c_tm1, ctx_tm1,
			  u_i, u_f, u_o, u_c, x_encoder, attention_encoder, x_img, B_W, B_U, B_Wimg, B_Wctx):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		attention_x = T.dot(x_t * B_W[4], self.W_x2a)						   # first part of attention function f x_t -> activation
		attention_total = attention_x[:,None,:] + attention_encoder	   # second part x_encoded -> activation
		if self.prev_context:										   # optional third part previous attended vector -> attention
			attention_prev = T.dot(ctx_tm1,self.W_ctx2a)
			attention_total += attention_prev[:,None,:]

		attention_activation = T.dot( T.tanh(attention_total), self.V) # attention -> scores
		attention_alpha = T.nnet.softmax(attention_activation[:,:,0])  # scores -> weights
		ctx_t = (x_encoder * attention_alpha[:,:,None]).sum(axis = 1)  # weighted average of context vectors

		xi_t = xi_t + T.dot(ctx_t * B_Wctx[0], self.Wctx_i)
		xf_t = xf_t + T.dot(ctx_t * B_Wctx[1], self.Wctx_f)
		xc_t = xc_t + T.dot(ctx_t * B_Wctx[2], self.Wctx_c)
		xo_t = xo_t + T.dot(ctx_t * B_Wctx[3], self.Wctx_o)

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * B_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * B_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * B_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * B_U[3], u_o))
		h_t = o_t * self.activation(c_t)

		pred1_t = T.dot(h_t, self.U_p1) + self.b_p1
		pred1_t = T.nnet.softmax(pred1_t.reshape((-1, pred1_t.shape[-1]))).reshape(pred1_t.shape)

		pred2_t = T.dot(h_t, self.U_p2) + self.b_p2
		pred2_t = T.nnet.softmax(pred2_t.reshape((-1, pred2_t.shape[-1]))).reshape(pred2_t.shape)

		pred3_t = T.dot(h_t, self.U_p3) + self.b_p3
		pred3_t = T.nnet.softmax(pred3_t.reshape((-1, pred3_t.shape[-1]))).reshape(pred3_t.shape)

		pred4_t = T.dot(h_t, self.U_p4) + self.b_p4
		pred4_t = T.nnet.softmax(pred4_t.reshape((-1, pred4_t.shape[-1]))).reshape(pred4_t.shape)

		return pred1_t, pred2_t, pred3_t, pred4_t, h_t, c_t, ctx_t

	def _step_test(self,
			  x_t, xi_t, xf_t, xo_t, xc_t, mask_tm1,
			  pred1_tm1, pred2_tm1, pred3_tm1, pred4_tm1, h_tm1, c_tm1, ctx_tm1, 
			  u_i, u_f, u_o, u_c, x_encoder, attention_encoder, x_img, B_W, B_U, B_Wimg, B_Wctx):

		outer1 = pred1_tm1[:, :, np.newaxis] * pred2_tm1[:, np.newaxis, :]
		outer1 =  outer1.reshape((outer1.shape[0],-1))
		outer2 = pred3_tm1[:, :, np.newaxis] * pred4_tm1[:, np.newaxis, :]
		outer2 =  outer2.reshape((outer2.shape[0],-1))
		pred = outer1[:, :, np.newaxis] * outer2[:, np.newaxis, :]
		pred =	pred.reshape((pred.shape[0],-1))
		x_t = self.W_embedding[T.argmax(pred, axis = 1)] * B_W[4]

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1

		attention_x = T.dot(x_t, self.W_x2a)
		attention_total = attention_x[:,None,:] + attention_encoder
		if self.prev_context:
			attention_prev = T.dot(ctx_tm1,self.W_ctx2a)
			attention_total += attention_prev[:,None,:]

		attention_activation = T.dot( T.tanh(attention_total), self.V) # attention -> scores
		attention_alpha = T.nnet.softmax(attention_activation[:,:,0])  # scores -> weights
		ctx_t = (x_encoder * attention_alpha[:,:,None]).sum(axis = 1)  # weighted average of context vectors

		xi_t = T.dot(x_t * B_W[0], self.W_i) + self.b_i + T.dot(x_img * B_Wimg[0], self.Wimg_i) + T.dot(ctx_t * B_Wctx[0], self.Wctx_i)
		xf_t = T.dot(x_t * B_W[1], self.W_f) + self.b_f + T.dot(x_img * B_Wimg[1], self.Wimg_f) + T.dot(ctx_t * B_Wctx[1], self.Wctx_f)
		xc_t = T.dot(x_t * B_W[2], self.W_c) + self.b_c + T.dot(x_img * B_Wimg[2], self.Wimg_c) + T.dot(ctx_t * B_Wctx[2], self.Wctx_c)
		xo_t = T.dot(x_t * B_W[3], self.W_o) + self.b_o + T.dot(x_img * B_Wimg[3], self.Wimg_o) + T.dot(ctx_t * B_Wctx[3], self.Wctx_o)

		i_t = self.inner_activation(xi_t + T.dot(h_mask_tm1 * B_U[0], u_i))
		f_t = self.inner_activation(xf_t + T.dot(h_mask_tm1 * B_U[1], u_f))
		c_t = f_t * c_mask_tm1 + i_t * self.activation(xc_t + T.dot(h_mask_tm1 * B_U[2], u_c))
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1 * B_U[3], u_o))
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

		return pred1_t, pred2_t, pred3_t, pred4_t, h_t, c_t, ctx_t

	def get_output(self, train = False, get_tuple = False):

		input_dict = self.get_input(train)
		X_idx = input_dict[self.dec_input_name]
		X = self.W_embedding[X_idx]
		X_img = input_dict[self.img_name]
		X_encoder = input_dict[self.enc_name]
		X_encoder = X_encoder.reshape((X_encoder.shape[0],X_encoder.shape[1],-1))
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		retain_prob = 1. - self.dropout

		if train:
			B_W = self.srng.binomial((5, X.shape[1], self.input_dim),  p=retain_prob, dtype=theano.config.floatX)
			B_U = self.srng.binomial((4, X.shape[1], self.hidden_dim), p=retain_prob, dtype=theano.config.floatX)
			B_Wctx = self.srng.binomial((4, X.shape[1], self.hidden_dim),  p=retain_prob, dtype=theano.config.floatX)
			B_Wimg = self.srng.binomial((4, X.shape[1], self.hidden_dim),  p=retain_prob, dtype=theano.config.floatX)
		else:
			B_W = np.ones(5, dtype=theano.config.floatX) * retain_prob
			B_U = np.ones(4, dtype=theano.config.floatX) * retain_prob
			B_Wctx = np.ones(4, dtype=theano.config.floatX) * retain_prob
			B_Wimg = np.ones(4, dtype=theano.config.floatX) * retain_prob

		xi = T.dot(X * B_W[0], self.W_i) + self.b_i + T.dot(X_img * B_Wimg[0], self.Wimg_i)[None,:,:]
		xf = T.dot(X * B_W[1], self.W_f) + self.b_f + T.dot(X_img * B_Wimg[1], self.Wimg_f)[None,:,:]
		xc = T.dot(X * B_W[2], self.W_c) + self.b_c + T.dot(X_img * B_Wimg[2], self.Wimg_c)[None,:,:]
		xo = T.dot(X * B_W[3], self.W_o) + self.b_o + T.dot(X_img * B_Wimg[3], self.Wimg_o)[None,:,:]

		attention_encoder = T.dot(X_encoder,self.W_e2a)
		if train:
			STEP = self._step
		else:
			STEP = self._step_test

		[outputs1, outputs2, outputs3, outputs4, hiddens, memories, ctx_vectors], updates = theano.scan(
			STEP,
			sequences=[X, xi, xf, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim),1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.hidden_dim),1),
			],
			non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c, X_encoder, attention_encoder, X_img, B_W, B_U, B_Wimg, B_Wctx],
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
				  "v_dim": self.v_dim,
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
				  "prev_context": self.prev_context,
				  "enc_name" : self.enc_name,
				  "dec_input_name" : self.dec_input_name,
				  "img_name" : self.img_name,
				  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
				  "U_regularizer": self.U_regularizer.get_config() if self.b_regularizer else None,
				  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
				  "dropout" : self.dropout
		}
		base_config = super(LSTMAttentionDecoder, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
