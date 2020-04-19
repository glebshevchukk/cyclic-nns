#adapted from https://github.com/guillaume-chevalier/Spiking-Neural-Network-SNN-with-PyTorch-where-Backpropagation-engenders-STDP/blob/master/Spiking%20Neural%20Networks%20with%20PyTorch.ipynb
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
import torch.functional as F
import copy

torch.autograd.set_detect_anomaly(True)

EVOL_STEPS=1000
N_CANDIDATES=32
NUM_PARENTS=16
OUTER_STEPS = 16
INNER_STEPS = 10
BATCH_SIZE=100
N_STATES=100
TEST_ITER=100
fig, (ax1, ax2) = plt.subplots(2, 1)
camera = Camera(fig)

#mask = ~torch.eye(N_STATES,dtype=bool)
image_mask = torch.ones((N_STATES,N_STATES))
image_mask[range(N_STATES),range(N_STATES)] = np.nan


class Neurons():
  def __init__(self, n_inputs,n_outputs,mask,n_hidden=N_STATES,decay_multiplier=0.9, threshold=0.01, penalty_threshold=0.05,n_timesteps=INNER_STEPS):
    assert n_inputs+n_outputs < n_hidden
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_outputs = n_outputs
    self.decay_multiplier = decay_multiplier
    self.threshold = threshold
    self.penalty_threshold = penalty_threshold
    self.n_timesteps=n_timesteps
    self.edge_weights = init_weights()
    self.mask = mask
    self.reset_state()

  
  def reset_state(self):
    self.prev_inner = torch.zeros([BATCH_SIZE,self.n_hidden])
    self.prev_outer = torch.zeros([BATCH_SIZE,self.n_hidden])

  def forward(self,x):
    input_excitation = x.matmul(self.edge_weights*self.mask)
    inner_excitation = input_excitation + self.prev_inner * self.decay_multiplier
    outer_excitation = torch.clamp(inner_excitation-self.threshold, 0.0)

    do_penalize_gate = (outer_excitation > 0).float()

    inner_excitation = inner_excitation - (self.penalty_threshold/self.threshold * inner_excitation) * do_penalize_gate

    outer_excitation = outer_excitation + torch.abs(self.prev_outer) * self.decay_multiplier / 2.0

    delayed_return_state = self.prev_inner
    delayed_return_output = self.prev_outer
    self.prev_inner = inner_excitation
    self.prev_outer = outer_excitation
    return delayed_return_output

  def full_pass(self,x):
    for _ in range(self.n_timesteps):
      x = self.forward(x)
    
    return x[:,-self.n_outputs]


class sin_loader:
  def __init_(self):
    pass
  def __iter__(self):
    return self
  def __next__(self):
    random_x = npr.uniform(low=-1.0, high=1.0, size=BATCH_SIZE)
    y = np.sin(random_x)
    return random_x, y


def init_weights():
  edge_weights = torch.randn((N_STATES,N_STATES),requires_grad=True)
  return edge_weights

def init_mask():
  mask = torch.FloatTensor(N_STATES, N_STATES).uniform_() > 0.3
  mask = mask.type(torch.int)
  return mask


def plot_states(params):
  states = params.detach()
  cmap = cm.get_cmap()
  cmap.set_bad(color='black')
  ax2.imshow(states.unsqueeze(0),cmap=cmap)
  camera.snap()

def plot_network(params):
  show_weights = params.detach()
  cmap = cm.get_cmap()
  cmap.set_bad(color='black')
  ax1.imshow(show_weights*image_mask,cmap=cmap)
  camera.snap()

def backprop_runner(neurons,loader):
  optimizer = optim.Adam([neurons.edge_weights])

  for n in range(OUTER_STEPS):
      optimizer.zero_grad()
      neurons.reset_state()
      train_states = torch.zeros((BATCH_SIZE,N_STATES))
      train_x,train_y=next(iter(loader))
      train_states[:,0] = torch.FloatTensor([train_x])
      train_y = torch.FloatTensor([train_y])
      
      y_pred = neurons.full_pass(train_states)
      l = (train_y - y_pred).pow(2).mean()
      #print(l)
     # if n % TEST_ITER== 0:
     #  plot_network(neurons.edge_weights)
     #  print(l)
      l.backward()
      optimizer.step()
    
  return l.detach().numpy()
  
  # animation = camera.animate()
  # animation.save('animation.mp4')

def evol_runner():
  loader = sin_loader()
  masks = [init_mask() for _ in range(N_CANDIDATES)]
  for _ in range(EVOL_STEPS):
    all_neurons = [Neurons(1,1,mask) for mask in masks]
    fitness = [backprop_runner(neurons,loader) for neurons in all_neurons]
    print(min(fitness)),
    parents = select_mating_pool(masks,fitness)
    offspring_crossover = perform_crossovers(parents)
    masks[0:NUM_PARENTS] = parents
    masks[NUM_PARENTS:] = offspring_crossover


#return the parents with the max fitness in order
def select_mating_pool(population,fitness):
  sorted_pop = [pop for _,pop in sorted(zip(fitness,population))]
  return sorted_pop[:NUM_PARENTS]
def perform_crossovers(parents):
  all_offspring=[]
  for (p1,p2) in zip(parents[:-1],parents[1:]):
    c = crossover(p1,p2)
    all_offspring.append(c)
  return all_offspring


#crossover: for each parent take first half of weights from first parent and second half from second parent
def crossover(parent_1,parent_2):
  lower_1 = torch.tril(parent_1)
  upper_2 = torch.triu(parent_2)
  new_mask = lower_1+upper_2
  mutation = torch.FloatTensor(N_STATES, N_STATES).uniform_() > 0.8
  mutation = mutation.type(torch.int)
  #for each offspring, add some small amount of random noise
  new_mask = new_mask+mutation
  return new_mask

evol_runner()
