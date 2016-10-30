# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..engine import Layer, InputSpec

class RAM(Layer):
	'''
	RAM
	-- regularize initial gx gy
	-- regularize final dx dy
	'''
	def __init__(self, width, height, t_glimpse = 8, output_dim = 128, read_n = 8, n_params = 6, debug_mode = False,
				 weights=None,
				 activation='tanh', init='uniform',inner_init='orthogonal', forget_bias_init='one', inner_activation='sigmoid',
				 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
				 W_constraint=None, b_constraint=None,
				 dropout_W=0., dropout_U=0., **kwargs):

		self.width	    = width
		self.height	    = height
		self.t_glimpse  = t_glimpse
		self.output_dim = output_dim
		self.read_n	    = read_n
		self.n_params   = n_params
		self.debug_mode = debug_mode

		self.activation			  = activations.get(activation)
		self.init				  = initializations.get(init)
		self.inner_init = initializations.get(inner_init)
		self.forget_bias_init = initializations.get(forget_bias_init)
		self.inner_activation = activations.get(inner_activation)

		self.dropout_W, self.dropout_U = dropout_W, dropout_U

		self.W_regularizer		  = regularizers.get(W_regularizer)
		self.b_regularizer		  = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.input_spec = [InputSpec(ndim=2)]

		self.initial_weights = weights
		if self.dropout_W or self.dropout_U:
			self.uses_learning_phase = True

		super(RAM, self).__init__(**kwargs)

	def build(self, input_shape):
		self.input_spec = [InputSpec(shape=input_shape)]
#		self.input_spec = [InputSpec(dtype=K.floatx(), shape=(None, self.width * self.height))]

		self.W_param  = self.init((self.output_dim, self.n_params), name='{}_W_param'.format(self.name))
		self.W = self.init((self.read_n* self.read_n, 4 * self.output_dim), name='{}_W'.format(self.name))
		self.U = self.inner_init((self.output_dim, 4 * self.output_dim),name='{}_U'.format(self.name))
		self.b = K.variable(np.hstack((np.zeros(self.output_dim),
									   K.get_value(self.forget_bias_init((self.output_dim,))),
									   np.zeros(self.output_dim),
									   np.zeros(self.output_dim))),
							name='{}_b'.format(self.name))


		self.b_param  = K.zeros((self.n_params,), name='{}_b_param'.format(self.name))
		self.h_init   = K.zeros((1,self.output_dim), name='{}_h_init'.format(self.name))
		self.c_init   = K.zeros((1,self.output_dim), name='{}_c_init'.format(self.name))

#		self.trainable_weights = [self.W, self.U, self.b, self.W_param, self.b_param, self.h_init, self.c_init]
		self.trainable_weights = [self.W, self.U, self.b, self.W_param, self.b_param]

		self.regularizers = []
		if self.W_regularizer:
			### check if all three should be set to the same var
			self.W_regularizer.set_param(self.W)
			self.W_regularizer.set_param(self.W_param)
			self.W_regularizer.set_param(self.U)
			self.regularizers.append(self.W_regularizer)

		if self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.b_regularizer.set_param(self.b_param)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if self.initial_weights is not None:
			### make sure you can save/load models
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.output_dim)

	def get_constants(self, x):
		constants = []
		if 0 < self.dropout_U < 1:
			ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, self.output_dim))
			B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
			constants.append(B_U)
		else:
			constants.append([K.cast_to_floatx(1.) for _ in range(4)])

		if 0 < self.dropout_W < 1:
#			input_shape = self.input_spec[0].shape
#			input_dim = input_shape[-1]
			ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
			ones = K.tile(ones, (1, self.read_n * self.read_n))
			B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
			constants.append(B_W)
		else:
			constants.append([K.cast_to_floatx(1.) for _ in range(4)])
		return constants

	def lstm_enc(self, x, states):
		h_tm1 = states[0]
		c_tm1 = states[1]
		B_U = states[2]
		B_W = states[3]

		z = K.dot(x * B_W[0], self.W) + K.dot(h_tm1 * B_U[0], self.U) + self.b
		z0 = z[:, :self.output_dim]
		z1 = z[:, self.output_dim: 2 * self.output_dim]
		z2 = z[:, 2 * self.output_dim: 3 * self.output_dim]
		z3 = z[:, 3 * self.output_dim:]

		i = self.inner_activation(z0)
		f = self.inner_activation(z1)
		c = f * c_tm1 + i * self.activation(z2)
		o = self.inner_activation(z3)

		h = o * self.activation(c)
		return h, [h, c]


	def filterbank(self, gx, gy, sigma2, delta_x, delta_y):

		grid_i = K.reshape(K.cast(K.range_( self.read_n), 'float32'), [1, -1])
		mu_x   = gx + (grid_i - self.read_n / 2 - 0.5) * delta_x # eq 19
		mu_y   = gy + (grid_i - self.read_n / 2 - 0.5) * delta_y # eq 20
		a	   = K.reshape(K.cast(K.range_( self.width ), 'float32'), [1, 1, -1])
		b	   = K.reshape(K.cast(K.range_( self.height), 'float32'), [1, 1, -1])
		mu_x   = K.reshape(mu_x, [-1, self.read_n, 1])
		mu_y   = K.reshape(mu_y, [-1, self.read_n, 1])
		sigma2 = K.reshape(sigma2, [-1, 1, 1])
		Fx	   = K.exp(-K.square((a - mu_x) / (2*sigma2))) # 2*sigma2?
		Fy	   = K.exp(-K.square((b - mu_y) / (2*sigma2))) # batch x N x B
		Fx	   = Fx/K.maximum(K.sum(Fx,2,keepdims=True),1e-8)
		Fy	   = Fy/K.maximum(K.sum(Fy,2,keepdims=True),1e-8)
		return Fx,Fy

	def attn_window(self, h_enc):
		params = K.dot(h_enc, self.W_param) + self.b_param
		gx_, gy_, log_sigma2, log_delta_x, log_delta_y, log_gamma = K.split(1,6,params)

		gx		 = (self.width	+1)/(2*(gx_+1))
		gy		 = (self.height +1)/(2*(gy_+1))
		sigma2	 = K.exp(log_sigma2)
		delta_x	 = ((max(self.width, self.height)-1)/( self.read_n -1))*K.exp(log_delta_x)
		delta_y	 = ((max(self.width, self.height)-1)/( self.read_n -1))*K.exp(log_delta_y)

		return self.filterbank(gx, gy, sigma2, delta_x, delta_y)+(K.exp(log_gamma), gx, gy, sigma2, delta_x, delta_y, params)

	def filter_img(self, img, Fx, Fy, gamma):
		Fxt		= K.permute_dimensions(Fx, pattern = [0,2,1])
		img		= K.reshape(img, [-1, self.height, self.width])
		glimpse = K.batch_dot(Fy,K.batch_dot(img,Fxt))
		glimpse = K.reshape(glimpse,[-1,self.read_n*self.read_n])
		return glimpse*K.reshape(gamma,[-1,1])

	def read(self, x, h_enc):
		Fx, Fy, gamma, gx, gy, sigma2, delta_x, delta_y, box_params = self.attn_window(h_enc)
		patch = self.filter_img(x, Fx, Fy, gamma)
		return patch, box_params

	def call(self, x, mask = None):

		if self.debug_mode:
			return self.get_boxes(x)
		input_shape = self.input_spec[0].shape

		constants	= self.get_constants(K.zeros((input_shape[0], self.read_n ** 2)))
#		output      = K.tile(self.h_init, K.pack([input_shape[0],1]))
#		ctm1        = K.tile(self.c_init, K.pack([input_shape[0],1]))
		output = K.zeros((input_shape[0], self.output_dim))
		ctm1   = K.zeros((input_shape[0], self.output_dim))

		for t in range(self.t_glimpse):
			patch, box_params = self.read(x, output)
			output, [ht, ctm1] =  self.lstm_enc(patch, [output, ctm1]+constants)
		### optionally add box parameters to output beware shape change in output
		return output

	def get_boxes(self, x):

		input_shape = self.input_spec[0].shape

		constants	= self.get_constants(K.zeros((input_shape[0], self.read_n ** 2)))
#		output      = K.tile(self.h_init, K.pack([input_shape[0],1]))
#		ctm1        = K.tile(self.c_init, K.pack([input_shape[0],1]))
		output = K.zeros((input_shape[0], self.output_dim))
		ctm1   = K.zeros((input_shape[0], self.output_dim))

		box_t   = []
		patch_t = []
		for t in range(self.t_glimpse):
			patch, box_params  = self.read(x, output)
			output, [ht, ctm1] = self.lstm_enc(patch, [output, ctm1]+constants)
			box_t             += [box_params]
			patch_t           += [patch]

		boxes   = K.concatenate(box_t, axis = 1)
		patches = K.concatenate(patch_t, axis = 1)
		output  = K.concatenate([output,boxes,patches], axis = 1)

		return output

	def get_config(self):
		config = { 'width'	   : self.width,
				   'height'	   : self.height,
				   't_glimpse' : self.t_glimpse,
				   'output_dim'	 : self.output_dim,
				   'read_n'	   : self.read_n,
				   'n_params'  : self.n_params,
				   'debug_mode': self.debug_mode,
				   'init': self.init.__name__,
				   'inner_init': self.inner_init.__name__,
				   'forget_bias_init': self.forget_bias_init.__name__,
				   'activation': self.activation.__name__,
				   'inner_activation': self.inner_activation.__name__,
				   'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
				   'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
				   'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
				   'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
				   'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
				   'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
				   'dropout_W': self.dropout_W,
				   'dropout_U': self.dropout_U}
		base_config = super(RAM, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
