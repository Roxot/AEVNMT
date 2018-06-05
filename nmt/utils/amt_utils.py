# Alternative MT utils.

import tensorflow as tf
import math

# Author: Wilker Aziz
def expand_into_2d_mask(repetitions, mask, dtype=tf.bool):
    """

    :param repetitions: N
    :param mask: [B, M]
    :param dtype: output type
    :return: [B, N, M]
    """
    valid = tf.cast(mask, dtype=dtype)
    # [B, N, M]
    valid = tf.tile(tf.expand_dims(valid, 1), [1, repetitions, 1])
    return valid

# Author: Wilker Aziz
def masked_softmax(logits, mask):
    logits = tf.where(
        condition=tf.cast(mask, tf.bool),
        x=logits,
        y=tf.fill(tf.shape(logits), float('-inf'))
    )
    cpd = tf.nn.softmax(logits)    
    return cpd

# Author: Wilker Aziz
def diagonal_attention_coefficients(x_mask, y_mask, x_len, y_len):

    longest_x = tf.shape(x_mask)[1]
    longest_y = tf.shape(y_mask)[1]

    # a trainable positive scalar
    diagonal_strength = tf.nn.softplus(tf.get_variable('diagonal_strength', initializer=0., dtype=tf.float32))

    # Then we compute event probabilities (cpd parameters)
    # [B]
    x_len = tf.cast(x_len, dtype=tf.float32)
    y_len = tf.cast(y_len, dtype=tf.float32)

    i = tf.cast(tf.zeros_like(x_mask, dtype=tf.int32) + tf.range(1, longest_x + 1), dtype=tf.float32)
    # normalised by source length
    norm_i = i / tf.expand_dims(x_len, 1)

    # Tensor of target positions
    # [B, N]
    j = tf.cast(tf.zeros_like(y_mask, dtype=tf.int32) + tf.range(1, longest_y + 1), dtype=tf.float32)
    norm_j = j / tf.expand_dims(y_len, 1)

    # Here we make a big tensor of absolute differences
    # [B, N, M]
    delta = tf.abs(tf.expand_dims(norm_i, 1) - tf.expand_dims(norm_j, 2))
    # and that makes the score of the diagonal
    logits = - delta * diagonal_strength

    # [B, N, M]
    cpd = masked_softmax(logits, expand_into_2d_mask(longest_y, x_mask))

    return cpd

# source: https://github.com/wilkeraziz/dgm4nlp
def self_attention_layer(
        inputs,
        num_steps,
        units,
        activation=tf.nn.softmax,
        mask_diagonal=False,
        mask_value=float('-inf'),
        name='self_attention',
        reuse=None):
  """
  Compute self attention levels (masking invalid positions).

  :param inputs: [batch_size, max_time, dim]
  :param num_steps: number of steps per training instance [batch_size]
  :param units: number of query/key units
  :param activation: defaults to tf.nn.softmax for normalised attention
  :param mask_diagonal: defaults to False
  :param mask_value: defaults to -inf
  :param name: defaults to self_attention
  :param reuse: passed to tf layers (defaults to None)
  :return: [batch_size, max_time, max_time]
  """
  batch_size = tf.shape(inputs)[0]  # B
  longest = tf.shape(inputs)[1]  # M
  with tf.variable_scope(name):
    # [B, M, d]
    queries = tf.layers.dense(inputs, units=units, name='queries', reuse=reuse)
    keys = tf.layers.dense(inputs, units=units, name='keys', reuse=reuse)
    # [B, M, M]
    scores = tf.matmul(
        queries,  # [B, M, d]
        keys,  # [B, M, d]
        transpose_b=True
    )
    # mask invalid logits
    # [B, M, M]
    condition = tf.tile(
        # make the boolean mask [B, 1, M]
        tf.expand_dims(
            # get a boolean mask [B, M]
            tf.sequence_mask(num_steps, maxlen=longest),
            1
        ),
        [1, longest, 1]
    )
    scores = tf.where(
        # make the boolean mask [B, M, M]
        condition=condition,
        x=scores,
        y=tf.ones(shape=[batch_size, longest, longest]) * mask_value
    )
    # mask diagonal
    if mask_diagonal:
      scores += tf.diag(tf.fill([tf.shape(scores)[-1]], mask_value))
    # Normalise attention
    # [B, M, M]
    #outputs = tf.where(
    #    condition=condition,
    #    x=activation(scores),
    #    y=tf.zeros_like(scores)
    #)
    return activation(scores)

# code from transformer
def add_timing_signal(x, min_timescale=1, max_timescale=1e4, num_timescales=16):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.
  This allows attention to learn to use absolute and relative positions.
  The timing signal should be added to some precursor of both the source
  and the target of the attention.
  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).
  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the depth dimension, padded with zeros to be the same depth as the input,
  and added into input.
  Source:
    https://github.com/tensorflow/tensor2tensor
  Args:
    x: a Tensor with shape [?, length, ?, depth]
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int <= depth/2
  Returns:
    a Tensor the same shape as x.
  """
  length = tf.shape(x)[1]
  depth = tf.shape(x)[3]
  signal = get_timing_signal(length, min_timescale, max_timescale,
                             num_timescales)
  padded_signal = tf.pad(signal, [[0, 0], [0, depth - 2 * num_timescales]])
  return x + tf.reshape(padded_signal, [1, length, 1, depth])

# code from transformer
# add position embeddings to your own embeddings
# suppose inputs is [batch_size, max_steps, emb_dim]
def enrich_embeddings_with_positions(inputs, units, name, reuse=None):
  with tf.variable_scope(name, reuse=reuse):
    num_timescales = units // 2
    inputs = tf.expand_dims(inputs, axis=2)
    inputs = add_timing_signal(inputs, num_timescales=num_timescales)
    inputs = tf.squeeze(inputs, axis=2)
    return inputs

# code from transformer
def get_timing_signal(length,
                      min_timescale=1,
                      max_timescale=1e4,
                      num_timescales=16):
  """Create Tensor of sinusoids of different frequencies.
  Args:
    length: Length of the Tensor to create, i.e. Number of steps.
    min_timescale: a float
    max_timescale: a float
    num_timescales: an int
  Returns:
    Tensor of shape (length, 2*num_timescales)
  """
  positions = tf.to_float(tf.range(length))
  log_timescale_increment = (
      math.log(max_timescale / min_timescale) / (num_timescales - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  scaled_time = tf.expand_dims(positions, 1) * tf.expand_dims(inv_timescales, 0)
  return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

# Source: https://github.com/wilkeraziz/dgm4nlp
class AnnealingSchedule:
  """
  This class implements helper code for an annealing schedule.
  """

  def __init__(self, initial=1., final=1., step=0.,
               wait=0, nb_updates=1, wait_value=None,
               step_fn=lambda alpha, step, final: min(alpha + step, final),
               name='alpha'):
    """
    :param initial:
    :param final:
    :param step:
    :param wait: how many updates should we wait before starting the schedule (the first step occurs after wait + nb_updates)
    :param nb_updates: number of updates between steps
    :param wait_value: a value (other than initial) to use while waiting
    :param step_fn: control a step in the schedule
        - the default step is additive and capped by `final`
        - one can design multiplicative steps
        - once can even make it a decreasing schedule
    :param name:
    """
    self._initial = initial
    self._final = final
    self._step = step
    self._wait = wait
    self._nb_updates = nb_updates
    self._alpha = initial
    self._alpha_while_waiting = wait_value if wait_value is not None else initial
    self._step_fn = step_fn
    self._counter = 0
    self.name = name

  def alpha(self):
    """Return the current alpha"""
    if self._wait > 0:
      return self._alpha_while_waiting
    return self._alpha

  def update(self):
    """
    Update schedule or waiting time.
    :param eoe: End-Of-Epoch flag
    :return: current alpha
    """
    if self._wait > 0:    # we are still waiting
      self._wait -= 1     # decrease the waiting time and keep waiting
      return self._alpha_while_waiting  # and keep waiting
    # We are done waiting, now we start counting
    self._counter += 1
    if self._counter < self._nb_updates:  # not enough updates
      return self._alpha
    else:  # enough updates, we are ready to reset the counter to zero 
      self._counter = 0
    # and apply a step 
    self._alpha = self._step_fn(self._alpha, self._step, self._final)
    return self._alpha
