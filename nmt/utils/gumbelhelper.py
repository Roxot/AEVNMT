import tensorflow as tf

from nmt.contrib.stat.dist import Gumbel

class GumbelHelper(tf.contrib.seq2seq.CustomHelper):

  def __init__(self, embedding_matrix, start_tokens, decode_lengths,
      straight_through=False):
    """Initializer.
    Args:
      embedding_matrix: The embedding matrix to use.
      start_tokens: `int32` vector shaped `[batch_size]`, the start tokens.
      decode_lengths: `int32` vector shaped `[batch_size]`, the length of the decoded sequences.
      straight_through: boolean, if True will use the argmax for the forward pass
      but gradients flow through the softmax vector.
    """
    self._gumbel = Gumbel()

    self._embedding_matrix = embedding_matrix
    self._vocab_size = embedding_matrix.shape[0]

    # Embed the start tokens.
    self._start_tokens = tf.convert_to_tensor(
        start_tokens, dtype=tf.int32, name="start_tokens")
    self._start_inputs = tf.nn.embedding_lookup(self._embedding_matrix,
        self._start_tokens)

    self._decode_lengths = tf.convert_to_tensor(decode_lengths,
        dtype=tf.int32, name="decode_lengths")
    self._straight_through = straight_through
    sample_ids_shape = tf.TensorShape([self._vocab_size])
    sample_ids_dtype = tf.float32

    # Call CustomHelper.__init__
    super(GumbelHelper, self).__init__(
        initialize_fn=self._initialize_fn,
        sample_fn=self._sample_fn,
        next_inputs_fn=self._next_inputs_fn,
        sample_ids_shape=sample_ids_shape,
        sample_ids_dtype=sample_ids_dtype)

  def _initialize_fn(self):
    finished = tf.tile([False], [tf.shape(self._start_tokens)[0]])
    return (finished, self._start_inputs)

  def _sample_fn(self, time, outputs, state):

    # The outputs should be logits.
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))

    # Use the output to sample from a Gumbel distribution.
    std_gumbel_sample = self._gumbel.random_standard(tf.shape(outputs))
    continuous_samples = tf.nn.softmax(outputs + std_gumbel_sample)

    if not self._straight_through:
      sample_ids = continuous_samples
    else:
      discrete_samples = tf.one_hot(tf.argmax(continuous_samples, axis=-1),
          self._vocab_size, dtype=tf.float32)
      sample_ids = tf.stop_gradient(discrete_samples - continuous_samples) \
          + continuous_samples

    return sample_ids

  def _next_inputs_fn(self, time, outputs, state, sample_ids):
    finished = (time + 1 >= self._decode_lengths)
    all_finished = tf.reduce_all(finished)

    # If we're finished the value of this doesn't matter, otherwise embed
    # the Gumbel samples by multiplying with the embedding matrix.
    next_inputs = tf.cond(
        all_finished,
        lambda: self._start_inputs,
        lambda: tf.matmul(sample_ids, self._embedding_matrix))
    return (finished, next_inputs, state)
