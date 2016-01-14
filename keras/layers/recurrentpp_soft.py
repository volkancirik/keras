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

class LSTMpp_soft(Recurrent):
    '''
    soft selection of gates
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
        super(LSTMpp_soft, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        scale=0.05

        self.W_g = self.init((input_dim, self.output_dim))
        self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, 9 , self.output_dim)))
        self.b_g = shared_zeros((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_g, self.U_g, self.b_g,
            self.W_c, self.U_c, self.b_c,
            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c):

        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1
        act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
        gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

        c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        ops = [c_mask_tm1,c_tilda,(c_mask_tm1 + c_tilda),T.maximum(c_mask_tm1, c_tilda),T.minimum(c_mask_tm1, c_tilda),c_mask_tm1 - c_tilda,c_mask_tm1 * c_tilda,0 * c_tilda,0 * c_tilda + 1]
        yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
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
            non_sequences=[self.U_g, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def _debug_step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, gates_tm1, u_g, u_o, u_c):

        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1
        act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
        gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

        c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
        ops = [c_mask_tm1,c_tilda,(c_mask_tm1 + c_tilda),T.maximum(c_mask_tm1, c_tilda),T.minimum(c_mask_tm1, c_tilda),c_mask_tm1 - c_tilda,c_mask_tm1 * c_tilda,0 * c_tilda,0 * c_tilda + 1]
        yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
        c_t = (gate.reshape((-1,gate.shape[-1])) * yshuff.reshape((-1,yshuff.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        gates_t = gate
        return h_t, c_t, gates_t

    def get_gates(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xg = T.dot(X, self.W_g) + self.b_g
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, gates], updates = theano.scan(
            self._debug_step,
            sequences=[xg, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim, 9), 1)
            ],
            non_sequences=[self.U_g, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        return gates, memories

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
        base_config = super(LSTMpp_soft, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LSTMmul_soft(Recurrent):
    '''
    soft selection of gates
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
        super(LSTMmul_soft, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[2]
        self.input = T.tensor3()

        scale=0.05

        self.W_g = self.init((input_dim, self.output_dim))
        self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, 3, self.output_dim)))
        self.b_g = shared_zeros((self.output_dim))

        self.W_c = self.init((input_dim, self.output_dim))
        self.U_c = self.inner_init((self.output_dim, self.output_dim))
        self.b_c = shared_zeros((self.output_dim))

        self.W_o = self.init((input_dim, self.output_dim))
        self.U_o = self.inner_init((self.output_dim, self.output_dim))
        self.b_o = shared_zeros((self.output_dim))

        self.params = [
            self.W_g, self.U_g, self.b_g,
            self.W_c, self.U_c, self.b_c,
            self.W_o, self.U_o, self.b_o,
        ]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c):

        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1
        act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
        gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

        ops = [c_mask_tm1,self.activation(xc_t + T.dot(h_mask_tm1, u_c)),c_mask_tm1 * self.activation(xc_t + T.dot(h_mask_tm1, u_c))]
        yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
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
            non_sequences=[self.U_g, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        if self.return_sequences:
            return outputs.dimshuffle((1, 0, 2))
        return outputs[-1]

    def _debug_step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, gates_tm1, u_g, u_o, u_c):

        h_mask_tm1 = mask_tm1 * h_tm1
        c_mask_tm1 = mask_tm1 * c_tm1
        act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
        gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

        ops = [c_mask_tm1,self.activation(xc_t + T.dot(h_mask_tm1, u_c)),c_mask_tm1 * self.activation(xc_t + T.dot(h_mask_tm1, u_c))]
        yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
        c_t = (gate.reshape((-1,gate.shape[-1])) * yshuff.reshape((-1,yshuff.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
        o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        gates_t = gate
        return h_t, c_t, gates_t


    def get_gates(self, train=False):
        X = self.get_input(train)
        padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
        X = X.dimshuffle((1, 0, 2))

        xg = T.dot(X, self.W_g) + self.b_g
        xc = T.dot(X, self.W_c) + self.b_c
        xo = T.dot(X, self.W_o) + self.b_o

        [outputs, memories, gates], updates = theano.scan(
            self._debug_step,
            sequences=[xg, xo, xc, padded_mask],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
                T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim, 3), 1)
            ],
            non_sequences=[self.U_g, self.U_o, self.U_c],
            truncate_gradient=self.truncate_gradient)

        return gates, memories


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
        base_config = super(LSTMmul_soft, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class LSTMkernel_soft(Recurrent):
	'''
	soft selection of gates
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
		super(LSTMkernel_soft, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()



		self.W_g = self.init((input_dim, self.output_dim))
#		self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, 6 , self.output_dim)))
		self.U_g = self.inner_init((self.output_dim, 6, self.output_dim))
		self.b_g = shared_zeros((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		scalar_init = 0.5
		scale=0.01

#		self.k_parameters = shared_ones((11,))
		self.k_parameters = sharedX(np.random.uniform(low=scalar_init-scale, high=scalar_init+scale, size=(11, )))
		# self.sigma_se = shared_scalar(scalar_init)
		# self.sigma_per = shared_scalar(scalar_init)
		# self.sigma_b_lin = shared_scalar(scalar_init)
		# self.sigma_v_lin = shared_scalar(scalar_init)
		# self.sigma_rq = shared_scalar(scalar_init)

		# self.l_se = shared_scalar(scalar_init)
		# self.l_per = shared_scalar(scalar_init)
		# self.l_lin = shared_scalar(scalar_init)
		# self.l_rq = shared_scalar(scalar_init)

		# self.alpha_rq = shared_scalar(scalar_init)
		# self.p_per = shared_scalar(scalar_init)

		self.params = [
			self.k_parameters,
#			self.sigma_se, self.sigma_per, self.sigma_b_lin, self.sigma_v_lin,self.sigma_rq,
#			self.l_se, self.l_per, self.l_lin, self.l_rq,
#			self.alpha_rq, self.p_per,
			self.W_g, self.U_g, self.b_g,
			self.W_c, self.U_c, self.b_c,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))

		sigma_se = self.k_parameters[0]
		sigma_per = self.k_parameters[1]
		sigma_b_lin = self.k_parameters[2]
		sigma_v_lin = self.k_parameters[3]
		sigma_rq = self.k_parameters[4]

		l_se = self.k_parameters[5]
		l_per = self.k_parameters[6]
		l_lin = self.k_parameters[7]
		l_rq = self.k_parameters[8]

		alpha_rq = self.k_parameters[9]
		p_per = self.k_parameters[10]

		k_se = T.pow(sigma_se,2) * T.exp( -T.pow(c_mask_tm1 - c_tilda,2) / (2* T.pow(l_se,2) ))
		k_per = T.pow(sigma_per,2) * T.exp( -2*T.pow(T.sin( math.pi*(c_mask_tm1 - c_tilda)/p_per ),2)  / ( T.pow(l_per,2) ))
		k_lin = T.pow(sigma_b_lin,2) + T.pow(sigma_v_lin,2)  * (c_mask_tm1 - l_lin) * (c_tilda - l_lin )
		k_rq = T.pow(sigma_rq,2) * T.pow( 1 + T.pow( (c_mask_tm1 - c_tilda),2)  / ( 2 * alpha_rq * T.pow(l_rq,2) ), -alpha_rq)

		ops = [c_mask_tm1,c_tilda,k_se, k_per, k_lin,k_rq]
		yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
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
			non_sequences=[self.U_g, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]


	def _debug_step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, gates_tm1, u_g, u_o, u_c):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		c_tilda = self.activation(xc_t + T.dot(h_mask_tm1, u_c))
		ops = [c_mask_tm1,c_tilda,(c_mask_tm1 + c_tilda),T.maximum(c_mask_tm1, c_tilda),T.minimum(c_mask_tm1, c_tilda),c_mask_tm1 - c_tilda,c_mask_tm1 * c_tilda,0 * c_tilda,0 * c_tilda + 1]
		yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
		c_t = (gate.reshape((-1,gate.shape[-1])) * yshuff.reshape((-1,yshuff.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)
		gates_t = gate
		return h_t, c_t, gates_t

	def get_gates(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xg = T.dot(X, self.W_g) + self.b_g
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories, gates], updates = theano.scan(
			self._debug_step,
			sequences=[xg, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim, 9), 1)
			],
			non_sequences=[self.U_g, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient)

		return gates, memories

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
		base_config = super(LSTMkernel_soft, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class LSTMbase_soft(Recurrent):
	'''
	soft selection of gates
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
		super(LSTMbase_soft, self).__init__(**kwargs)

	def build(self):
		input_dim = self.input_shape[2]
		self.input = T.tensor3()

		scale=0.05

		self.W_g = self.init((input_dim, self.output_dim))
		self.U_g = sharedX(np.random.uniform(low=-scale, high=scale, size=(self.output_dim, 2, self.output_dim)))
		self.b_g = shared_zeros((self.output_dim))

		self.W_c = self.init((input_dim, self.output_dim))
		self.U_c = self.inner_init((self.output_dim, self.output_dim))
		self.b_c = shared_zeros((self.output_dim))

		self.W_o = self.init((input_dim, self.output_dim))
		self.U_o = self.inner_init((self.output_dim, self.output_dim))
		self.b_o = shared_zeros((self.output_dim))

		self.params = [
			self.W_g, self.U_g, self.b_g,
			self.W_c, self.U_c, self.b_c,
			self.W_o, self.U_o, self.b_o,
		]

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def _step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, u_g, u_o, u_c):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)

		ops = [c_mask_tm1,self.activation(xc_t + T.dot(h_mask_tm1, u_c))]
		yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
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
			non_sequences=[self.U_g, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient)

		if self.return_sequences:
			return outputs.dimshuffle((1, 0, 2))
		return outputs[-1]

	def _debug_step(self,xg_t, xo_t, xc_t, mask_tm1,h_tm1, c_tm1, gates_tm1, u_g, u_o, u_c):

		h_mask_tm1 = mask_tm1 * h_tm1
		c_mask_tm1 = mask_tm1 * c_tm1
		act = T.tensordot( xg_t + h_mask_tm1, u_g , [[1],[2]])
		gate = T.nnet.softmax(act.reshape((-1, act.shape[-1]))).reshape(act.shape)


		ops = [c_mask_tm1,self.activation(xc_t + T.dot(h_mask_tm1, u_c))]
		yshuff = T.as_tensor_variable( ops, name='yshuff').dimshuffle(1,2,0)
		c_t = (gate.reshape((-1,gate.shape[-1])) * yshuff.reshape((-1,yshuff.shape[-1]))).sum(axis = 1).reshape(gate.shape[:2])
		o_t = self.inner_activation(xo_t + T.dot(h_mask_tm1, u_o))
		h_t = o_t * self.activation(c_t)

#		gates_t = gate.reshape((gate.shape[0],gate.shape[1]*gate.shape[2]))
		gates_t = gate
		return h_t, c_t, gates_t

	def get_gates(self, train=False):
		X = self.get_input(train)
		padded_mask = self.get_padded_shuffled_mask(train, X, pad=1)
		X = X.dimshuffle((1, 0, 2))

		xg = T.dot(X, self.W_g) + self.b_g
		xc = T.dot(X, self.W_c) + self.b_c
		xo = T.dot(X, self.W_o) + self.b_o

		[outputs, memories, gates], updates = theano.scan(
			self._debug_step,
			sequences = [xg, xo, xc, padded_mask],
			outputs_info=[
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim), 1),
				T.unbroadcast(alloc_zeros_matrix(X.shape[1], self.output_dim, 2), 1)
			],
			non_sequences=[self.U_g, self.U_o, self.U_c],
			truncate_gradient=self.truncate_gradient)

		return gates, memories


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
		base_config = super(LSTMbase_soft, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
