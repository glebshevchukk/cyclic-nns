from __future__ import print_function
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import random
import itertools
import numpy as np
import numpy.random as npr
import torch
from celluloid import Camera
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)


OUTER_STEPS = 1000
INNER_STEPS = 4
BATCH_SIZE=20
N_STATES=4
TEST_ITER=10
fig, (ax1, ax2) = plt.subplots(2, 1)
camera = Camera(fig)

mask = ~torch.eye(N_STATES,dtype=bool)
image_mask = torch.ones((N_STATES,N_STATES))
image_mask[range(N_STATES),range(N_STATES)] = np.nan

class sin_loader:
  def __init_(self):
    pass
  def __iter__(self):
    return self
  def __next__(self):
    random_x = npr.uniform(low=-1.0, high=1.0, size=BATCH_SIZE)
    y = np.cos(random_x)+np.tan(random_x)
    return random_x, y


def forward(state,edge_weights,state_bias):
    for step in range(INNER_STEPS):
        state = state.matmul(edge_weights)
        state += state_bias
        #state = torch.sigmoid(state)
        #state = state.clamp(min=0)
    return state[:,-1]

def loss(state,y_true,edge_weights,state_bias):
    y_pred = forward(state,edge_weights,state_bias)
    return (y_true - y_pred).pow(2).mean()


def init():
  edge_weights = torch.randn((N_STATES,N_STATES),requires_grad=True)
  state_bias = torch.randn((N_STATES),requires_grad=True)
  return [edge_weights,state_bias]


def plot_network(params):
  show_weights = params[0].detach()
  cmap = cm.get_cmap()
  cmap.set_bad(color='black')
  ax1.imshow(show_weights*image_mask,cmap=cmap)
  show_bias = np.expand_dims(params[1].detach(),0)

  ax2.imshow(show_bias)
  camera.snap()

def backprop_runner():
  loader = sin_loader()
  edge_weights,state_bias=init()
  optimizer = optim.Adam([edge_weights,state_bias])

  for n in range(OUTER_STEPS):
      optimizer.zero_grad()
      train_states = torch.zeros((BATCH_SIZE,N_STATES))
      train_x,train_y=next(iter(loader))
      train_states[:,0] = torch.FloatTensor([train_x])
      train_y = torch.FloatTensor([train_y])
      
      l = loss(train_states,train_y,edge_weights,state_bias)
      if n % TEST_ITER== 0:
        plot_network([edge_weights,state_bias])
        print(l)
      l.backward()
      optimizer.step()
  
  animation = camera.animate()
  animation.save('animation.mp4')

backprop_runner()

