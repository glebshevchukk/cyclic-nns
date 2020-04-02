from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
import random
import itertools
import jax
from jax import grad
import jax.numpy as np
from celluloid import Camera
import numpy as onp
import numpy.random as onpr


OUTER_STEPS = 2000
INNER_STEPS = 32
BATCH_SIZE=32
N_STATES=4
NUM_BATCHES=4
LR=1.0
TEST_ITER=100

fig, (ax1, ax2) = plt.subplots(2, 1)
camera = Camera(fig)

class sin_loader:
  def __init_(self):
    pass
  def __iter__(self):
    return self
  def __next__(self):
    random_x = onpr.uniform(low=-1.0, high=1.0, size=BATCH_SIZE)
    y = np.sin(random_x)
    return random_x, y

# Sigmoid nonlinearity
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
  return np.clip(x,0)

# l2 loss
def l2_loss(params,states,y):
    edge_weights,state_bias = params
    for n in range(INNER_STEPS):
        states = step(states,edge_weights,state_bias)
    out = states[-1]
    l2 = (y-out)**2
    return l2


def init():
  states = onp.zeros((N_STATES))
  edge_weights = onpr.uniform(low=-1.0,high=1.0,size=(N_STATES,N_STATES))
  state_bias = onpr.uniform(low=-1.0,high=1.0,size=(N_STATES))
  return [states, edge_weights,state_bias]

def step(states,edge_weights,state_bias):
  prod = np.dot(states,edge_weights)
  prod += state_bias
  f = sigmoid(prod)
  return f

def plot_network(params):
  show_weights = onp.asarray(params[0])
  ax1.imshow(show_weights)
  show_bias = np.expand_dims(onp.asarray(params[1]),0)
  ax2.imshow(show_bias)
  camera.snap()

def backprop_runner():
  loader = sin_loader()
  states,edge_weights,state_bias=init()

  loss_grad = jax.jit(jax.vmap(jax.grad(l2_loss), in_axes=(None, 0, 0), out_axes=0))
  test_loss_func = jax.vmap(l2_loss, in_axes=(None, 0, 0), out_axes=0)

  itercount = itertools.count()
  for n in range(OUTER_STEPS):
      train_states = onp.zeros((BATCH_SIZE,N_STATES))
      train_x,train_y=next(iter(loader))
      train_states[:,0] = train_x

      grads = loss_grad([edge_weights,state_bias],train_states,train_y)
      #actual update over minibatch
      edge_weights -= LR*np.mean(grads[0],axis=0)
      state_bias -= LR*np.mean(grads[1],axis=0)

      if n % TEST_ITER == 0:
          plot_network([edge_weights,state_bias])
          test_x,test_y = next(iter(loader))
          test_states = onp.zeros((BATCH_SIZE,N_STATES))
          test_states[:,0] = test_x
          test_loss_val = test_loss_func([edge_weights,state_bias],test_states,test_y)

          print(f"Test loss for iter {n}: {test_loss_val.mean()}")
  

  animation = camera.animate()
  animation.save('animation.mp4')

backprop_runner()

