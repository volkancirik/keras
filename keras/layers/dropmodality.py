# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

import theano
import theano.tensor as T
import numpy as np

from ..layers.core import Layer
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams

class DropModality(Layer):
	'''
	drop a modality alltogether
	'''
	def __init__(self, layers,concat_axis = -1, **kwargs):
		if len(layers) < 2:
			raise Exception("Please specify two or more input layers (or containers) to merge")

		self.trng = RandomStreams(seed=np.random.randint(10e6))
		input_shapes = set()
		concat_axis = -1
		for l in layers:
			oshape = list(l.output_shape)
			oshape.pop(concat_axis)
			oshape = tuple(oshape)
			input_shapes.add(oshape)
		if len(input_shapes) > 1:
			raise Exception("dropmodality can only merge layers with matching output shapes except for the concat axis. Layer shapes: %s" % ([l.output_shape for l in layers]))

		self.layers = layers
		self.params = []
		self.regularizers = []
		self.constraints = []
		self.updates = []
		self.concat_axis = concat_axis
		for l in self.layers:
			params, regs, consts, updates = l.get_params()
			self.regularizers += regs
			self.updates += updates
			# params and constraints have the same size
			for p, c in zip(params, consts):
				if p not in self.params:
					self.params.append(p)
					self.constraints.append(c)

	@property
	def output_shape(self):
		input_shapes = [layer.output_shape for layer in self.layers]
		output_shape = list(input_shapes[0])
		for shape in input_shapes[1:]:
			output_shape[self.concat_axis] += shape[self.concat_axis]
		return tuple(output_shape)

	def get_params(self):
		return self.params, self.regularizers, self.constraints, self.updates


	def get_output(self, train=False):
		inputs = [self.layers[i].get_output(train) for i in range(len(self.layers))]
		output = T.concatenate(inputs, axis= self.concat_axis)

		full = T.concatenate([T.ones_like(inputs[i]) for i in xrange(len(inputs))], axis = self.concat_axis)
		masks = [full]

		for i in xrange(len(inputs)):
			outlist = []
			for j in xrange(len(inputs)):
				if i == j:
					mask = T.zeros_like(inputs[j])
				else:
					mask = T.ones_like(inputs[j])
				outlist += [mask]
			masks += [T.concatenate(outlist, axis = self.concat_axis)]

		masked = T.stack(masks)
		if train:
			index  = self.trng.random_integers(size=(1,),low = 0, high = len(masks)-1)[0]
		else:
			index = 0
		masked_output = output * masked[index]

		return masked_output

	def get_input(self, train=False):
		res = []
		for i in range(len(self.layers)):
			o = self.layers[i].get_input(train)
			if not type(o) == list:
				o = [o]
			for output in o:
				if output not in res:
					res.append(output)
		return res

	@property
	def input(self):
		return self.get_input()

	def supports_masked_input(self):
		return False

	def get_output_mask(self, train=None):
		return None

	def get_weights(self):
		weights = []
		for l in self.layers:
			weights += l.get_weights()
		return weights

	def set_weights(self, weights):
		for i in range(len(self.layers)):
			nb_param = len(self.layers[i].params)
			self.layers[i].set_weights(weights[:nb_param])
			weights = weights[nb_param:]

	def get_config(self):
		config = {"name": self.__class__.__name__,
				  "layers": [l.get_config() for l in self.layers],
				  "concat_axis": self.concat_axis
				  }
		base_config = super(DropModality, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
