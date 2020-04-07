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
import jax.random as npr
from jax.ops import index, index_add, index_update
from jax import lax
from jax.experimental import loops



OUTER_STEPS = 100
INNER_STEPS = 6
BATCH_SIZE=128
N_STATES=3
LR=0.1
TEST_ITER=10
NUM_PARENTS=10
POP_SIZE=10
fig, (ax1, ax2) = plt.subplots(2, 1)
camera = Camera(fig)

class sin_loader:
  def __init_(self):
    pass
  def __iter__(self):
    return self
  def __next__(self):
    random_x = onpr.uniform(low=-1.0, high=1.0, size=BATCH_SIZE)
    y = random_x
    return random_x, y

# Sigmoid nonlinearity
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
  return np.clip(x,0)

# l2 loss
def l2_loss(params,states,y):
    edge_weights,state_bias = params
    mask = ~np.eye(N_STATES,dtype=bool)
    for step in range(INNER_STEPS):
        states = np.matmul(states,edge_weights*mask)
        states += state_bias
        states = np.clip(states,0)
      
    out = states[-1]
    l2 = (y-out)**2
    return l2


def init():
  states = onp.zeros((N_STATES))
  edge_weights = onpr.uniform(low=-1.0,high=1.0,size=(N_STATES,N_STATES))
  state_bias = onpr.uniform(low=-1.0,high=1.0,size=(N_STATES))
  return [states, edge_weights,state_bias]

def genetic_init():
  states = onp.zeros((POP_SIZE,N_STATES))
  edge_weights = onpr.uniform(low=-1.0,high=1.0,size=(POP_SIZE,N_STATES,N_STATES))
  state_bias = onpr.uniform(low=-1.0,high=1.0,size=(POP_SIZE,N_STATES))
  return [states, edge_weights,state_bias]


def plot_network(params):
  show_weights = onp.asarray(params[0])
  ax1.imshow(show_weights)
  show_bias = np.expand_dims(onp.asarray(params[1]),0)
  ax2.imshow(show_bias)
  camera.snap()

def backprop_runner():
  loader = sin_loader()
  states,edge_weights,state_bias=init()

  loss_grad = jax.vmap(jax.grad(l2_loss), in_axes=(None, 0, 0), out_axes=0)
  test_loss_func = jax.vmap(l2_loss, in_axes=(None, 0, 0), out_axes=0)

  for n in range(OUTER_STEPS):
      train_states = onp.zeros((BATCH_SIZE,N_STATES))
      train_x,train_y=next(iter(loader))
      train_states[:,0] = train_x

      grads = loss_grad([edge_weights,state_bias],train_states,train_y)
      #print(grads[1])
      #actual update over minibatch
      edge_weights -= LR*np.mean(grads[0],axis=0)
      state_bias -= LR*np.mean(grads[1],axis=0)

      if n % TEST_ITER == 0:
          plot_network([edge_weights,state_bias])
          print(np.mean(grads[0],axis=0))
          test_x,test_y = next(iter(loader))
          test_states = onp.zeros((BATCH_SIZE,N_STATES))
          test_states[:,0] = test_x
          test_loss_val = test_loss_func([edge_weights,state_bias],test_states,test_y)

          print(f"Test loss for iter {n}: {test_loss_val.mean()}")
  
  animation = camera.animate()
  animation.save('animation.mp4')




#return the parents with the max fitness in order
def select_mating_pool(population,fitness,num):
  sorted_args = fitness.argsort()
  sorted_pop = population[sorted_args]
  return sorted_pop[:num]
#crossover: for each parent take first half of weights from first parent and second half from
#second parent
def crossover(parent_1,parent_2, offspring_size):
  all_offspring = []
  for o in range(offspring_size):
    lower_1 = np.tril(parent_1)
    upper_2 = np.triu(parent_2)
    offspring = lower_1+upper_2
    all_offspring.append(offspring)

#for each offspring, add some small amount of random noise
def mutation(offspring):
  return offspring + 0.1*onpr.uniform(-1.,1.,size=offspring.shape)

# def genetic_runner():
#   loader = sin_loader()
#   states,edge_weights,state_bias=genetic_init()
#   #for training, we're also vectorizing over the possible edge weights and state_biases
#   train_loss_func = jax.vmap(l2_loss, in_axes=(0, 0, 0), out_axes=0)
#   test_loss_func = jax.vmap(l2_loss, in_axes=(None, 0, 0), out_axes=0)

#   for n in range(OUTER_STEPS):
#       train_states = onp.zeros((BATCH_SIZE,N_STATES))
#       train_x,train_y=next(iter(loader))
#       train_states[:,0] = train_x
#       fitness = train_loss_func([edge_weights,state_bias],train_states,train_y)

#       parents = select_mating_pool([edge_weights,state_bias], fitness, NUM_PARENTS)
 
#       # Generating next generation using crossover.
#       offspring_crossover = crossover(parents,
#                                         offspring_size=(pop_size[0]-parents.shape[0], num_weights))
 
#       # Adding some variations to the offsrping using mutation.
#       offspring_mutation = mutation(offspring_crossover)

#       new_population[0:parents.shape[0], :] = parents
#       new_population[parents.shape[0]:, :] = offspring_mutation

#   animation = camera.animate()
#   animation.save('animation.mp4')

backprop_runner()

