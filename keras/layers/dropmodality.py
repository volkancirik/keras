# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros, shared_ones
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

class DropModality(Layer):
	'''
	drop a modality alltogether
	'''
	def __init__(self, input_shapes = [], **kwargs):
		self.trng = RandomStreams(seed=np.random.randint(10e6))
		self.params = []
		self.input_shapes = input_shapes

	def set_prev_shape(self, input_shapes):
		self.input_shapes = input_shapes

	def get_output(self, train=False):
		X = self.get_input(train)

		full = T.ones_like(X)
		masks = [full]

		for i in xrange(len(self.input_shapes)):
			mask = T.ones_like(X)
			idx = 0
			for j in xrange(len(self.input_shapes)):
				if i == j:
					mask = T.set_subtensor(mask[:,:,idx : idx+ self.input_shapes[j]], 0)
				idx =  idx + self.input_shapes[j]
			masks += [mask]
		masked = T.stack(masks)
		if train:
			index  = self.trng.random_integers(size=(1,),low = 0, high = len(masks)-1)[0]
		else:
			index = 0
		masked_output = X * masked[index]
		return masked_output

	def get_masked(self, train=False):
		X = self.get_input(train)

		full = T.ones_like(X)
		masks = [full]

		for i in xrange(len(self.input_shapes)):
			mask = T.ones_like(X)
			idx = 0
			for j in xrange(len(self.input_shapes)):
				if i == j:
					mask = T.set_subtensor(mask[:,:,idx : idx+ self.input_shapes[j]], 0)
				idx =  idx + self.input_shapes[j]
			masks += [mask]
		masked = T.stack(masks)
		index  = self.trng.random_integers(size=(1,),low = 0, high = len(masks)-1)[0]

		return masked, index

	def get_input_shapes(self):
		return self.input_shapes

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "input_shapes" : self.input_shapes
				  }
		base_config = super(DropModality, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
