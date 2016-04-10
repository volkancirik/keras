from __future__ import absolute_import
import theano.tensor as T


def softmax(x):
	return T.nnet.softmax(x.reshape((-1, x.shape[-1]))).reshape(x.shape)


def time_distributed_softmax(x):
	import warnings
	warnings.warn("time_distributed_softmax is deprecated. Just use softmax!", DeprecationWarning)
	return softmax(x)

def softplus(x):
	return T.nnet.softplus(x)


def relu(x):
	return T.nnet.relu(x)


def tanh(x):
	return T.tanh(x)


def sigmoid(x):
	return T.nnet.sigmoid(x)


def hard_sigmoid(x):
	return T.nnet.hard_sigmoid(x)


def linear(x):
	'''
	The function returns the variable that is passed in, so all types work
	'''
	return x

def stair(x, p = 3):
	x = T.floor(x * 1000) / 1000
	return T.floor(x+0.5) + 0.5 * ((2*(x - T.floor(x+0.5)))**p)

def rmax(x):

	xmax  = T.ge(x, T.max(x, axis = 1).reshape((x.shape[0],1)))
	shift = (T.ones_like(x) - xmax) * x
	max2  = T.max(shift,axis = 1).reshape((x.shape[0],1))
	out = T.nnet.relu(x - max2)

	return out

from .utils.generic_utils import get_from_module
def get(identifier):
	return get_from_module(identifier, globals(), 'activation function')
