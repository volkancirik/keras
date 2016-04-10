# -*- coding: utf-8 -*-
from __future__ import absolute_import, division
import theano.tensor as T
import numpy as np

from ..layers.core import MaskedLayer
from ..layers.core import Layer

class Average(MaskedLayer):
	def __init__(self, **kwargs):
		super(Average, self).__init__(**kwargs)

	def get_output(self, train=False):
		X = self.get_input(train)
		return T.mean(X,axis = 1)

	@property
	def output_shape(self):
		input_shape = self.input_shape
		return (input_shape[0], input_shape[2])


	def get_config(self):
		config = {"name": self.__class__.__name__}
		base_config = super(Average, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def get_output_mask(self, train=None):
		return None


# class Average(Layer):
#	def __init__(self, **kwargs):
#		super(Average, self).__init__(**kwargs)

#	def get_output(self, train=False):
#		X = self.get_input(train)
#		return X

#	def get_config(self):
#		config = {"name": self.__class__.__name__
#				  }
#		base_config = super(Average, self).get_config()
#		return dict(list(base_config.items()) + list(config.items()))
