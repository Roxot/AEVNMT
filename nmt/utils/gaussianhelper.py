import tensorflow as tf

class GaussianHelper(tf.contrib.seq2seq.CustomHelper):

  def __init__(self, start_tokens, decode_lengths, full_covariance=False):
    """Initializer.
    Args:
      start_tokens: `int32` vector shaped `[batch_size, num_emb_units]`, the start tokens.
      decode_lengths: `int32` vector shaped `[batch_size]`, the length of the decoded sequences.
    """
    self._num_emb_units = start_tokens.shape[1]
    self._batch_size = tf.shape(start_tokens)[0]
    self._start_tokens = start_tokens
    self._full_covariance = full_covariance

    # Embed the start tokens.
    self._decode_lengths = tf.convert_to_tensor(decode_lengths,
        dtype=tf.int32, name="decode_lengths")
    sample_ids_shape = tf.TensorShape([self._num_emb_units])
    sample_ids_dtype = tf.float32

    # Call CustomHelper.__init__
    super(GaussianHelper, self).__init__(
        initialize_fn=self._initialize_fn,
        sample_fn=self._sample_fn,
        next_inputs_fn=self._next_inputs_fn,
        sample_ids_shape=sample_ids_shape,
        sample_ids_dtype=sample_ids_dtype)

  def _initialize_fn(self):
    finished = tf.tile([False], [tf.shape(self._start_tokens)[0]])
    return (finished, self._start_tokens)

  def _sample_fn(self, time, outputs, state):

    # The outputs should be logits.
    if not isinstance(outputs, tf.Tensor):
      raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                      type(outputs))

    mean = outputs[:, :self._num_emb_units]

    if not self._full_covariance:
      stddev = outputs[:, self._num_emb_units:]
      sample_ids = mean + tf.random_normal(tf.shape(stddev)) * stddev
    else:

      # Predict the cholesky factor.
      cov_matrix_values = outputs[:, self._num_emb_units:]
      cov_matrix = tf.reshape(cov_matrix_values,
          [self._batch_size, self._num_emb_units, self._num_emb_units])
      cholesky = tf.contrib.distributions.matrix_diag_transform(cov_matrix,
          transform=tf.nn.softplus)
      mvn = tf.contrib.distributions.MultivariateNormalTriL(
          loc=mean, scale_tril=cholesky)
      sample_ids = mvn.sample()

    return sample_ids

  def _next_inputs_fn(self, time, outputs, state, sample_ids):
    finished = (time + 1 >= self._decode_lengths)
    all_finished = tf.reduce_all(finished)

    next_inputs = tf.cond(
        all_finished,
        lambda: self._start_tokens,
        lambda: sample_ids)
    return (finished, next_inputs, state)
