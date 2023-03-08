from functools import wraps
import time
import haiku as hk
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import io
import tensorflow as tf
import jax.tree_util as tree
import os


def print_pytree(v, name = "trainable variables"):
    print('==================================================================')
    print('# {}:'.format(name), sum(x.size for x in tree.tree_leaves(v)), flush=True)
    shape = tree.tree_map(lambda x: x.shape, v)
    for key in shape:
      print(key, shape[key])
    print('# {}:'.format(name), sum(x.size for x in tree.tree_leaves(v)), flush=True)
    print('==================================================================')


class MLPLN(hk.Module):
  def __init__(self, widths, name = None, layer_norm=False):
    '''MLP with layer normalization'''
    super().__init__(name)
    self.network = hk.nets.MLP(widths, activate_final=False)
    if layer_norm:
      self.network = hk.Sequential([self.network, 
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)])
  def __call__(self,x):
    return self.network(x)
    


def timeit_full(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper


def get_train_iter(params, loss_fun, optimizer = None):
  if optimizer is None:
    optimizer = optax.adam(learning_rate=1e-4)
  opt_state = optimizer.init(params)
  d_loss_d_theta = jax.jit(jax.grad(loss_fun))

  # @jax.jit
  def train_iter(params, rng_key, opt_state, *args):
    grads = d_loss_d_theta(params, rng_key, *args)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
  
  return opt_state, train_iter


def get_partial_optimizer(params, trainable_key_list, untrainable_key_list):
  def is_in_list(key):
    flag = "zero"
    for substr in trainable_key_list:
      if substr in key:
        flag="adamw"
    for substr in untrainable_key_list:
      if substr in key:
        flag="zero"
    return flag
  
  param_labels = {}
  for key, value_dict in params.items():
    param_labels[key] = {k: is_in_list(key) for k in value_dict}

  for key in param_labels:
    print(key, param_labels[key])
  optimizer = optax.multi_transform(
        {'adamw': optax.adam(learning_rate=1e-4), 'zero': optax.set_to_zero()}, param_labels)
  return optimizer

def get_train_iter_batch(params, loss_fn_batch, optimizer = None, averaged = True):
  if optimizer is None:
    optimizer = optax.adam(learning_rate=1e-4)
  opt_state = optimizer.init(params)
  if not averaged:
    loss_fn_mean = lambda *args: jnp.mean(loss_fn_batch(*args))
  else:
    loss_fn_mean = loss_fn_batch
  # @jax.jit
  def train_iter(params, rng_key, opt_state, *args):
    grads = jax.grad(loss_fn_mean)(params, rng_key, *args)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state
  
  return opt_state, train_iter


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


class DataSet():
  '''
  DataSet for training
  '''
  def __init__(self, data_list, buffer_cap, key):
    '''
    data_list: a list of arrays, with the same size in axis = 0
              elements are different components of data, e.g. u, left bound, right bound 
              should be aligned
    buffer_cap: the capacity of buffer
    '''
    self.key = key
    self.buffer = data_list
    self.buffer_cap = buffer_cap
    self.build_data()

  def add_to_buffer(self, new_data_list):
    buffer = [jnp.concatenate([d, nd], axis = 0)[-self.buffer_cap:,...] for d, nd in zip(self.buffer, new_data_list)]
    self.buffer = buffer
    self.build_data()

  def build_data(self):
    self.key, subkey = jax.random.split(self.key)
    # same random key, same permutation
    self.data = [jax.random.permutation(subkey, d, axis=0) for d in self.buffer]
    self.pointer = 0
    self.size = len(self.data[0])
    print("current buffer size: {}/{}, data size: {}".format(
                    len(self.buffer[0]), self.buffer_cap, self.size), flush=True)
  
  def next(self, bs):
    if self.pointer + bs > self.size:
      self.build_data()
    out = [d[self.pointer:self.pointer + bs, ...] for d in self.data]
    self.pointer += bs
    return out

if __name__ == "__main__":
  key = jax.random.PRNGKey(42)
  x = jax.random.uniform(key, (100,40))
  y = jnp.sum(x, axis = 1)
  
  dataset = DataSet([x,y], 150, jax.random.PRNGKey(0))
  dataset.add_to_buffer([x*1.1,y*1.1])
  xp, yp = dataset.next(20)
  assert jnp.sum((jnp.sum(xp, axis=1) - yp)**2) < 1e-6
  dataset.add_to_buffer([x*2,y*2])
  xp, yp = dataset.next(20)
  assert jnp.sum((jnp.sum(xp, axis=1) - yp)**2) < 1e-6
  print_pytree({"a":{"w":jnp.ones((1,2)), "b":jnp.ones((3,4))}})