"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt import model_helper
from . import DSimpleJointModel

class DVAEJointModel(DSimpleJointModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None, no_summaries=False):

    super(DVAEJointModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args, no_summaries=True)

    # Set model specific training summaries.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN and not no_summaries:
      self.bi_summary = tf.summary.merge([
          self._base_summaries,
          tf.summary.scalar("supervised_tm_accuracy", self._tm_accuracy),
          tf.summary.scalar("supervised_ELBO", self._elbo),
          tf.summary.scalar("supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("supervised_KL_Z", self._KL_Z),
          tf.summary.scalar("supervised_lm_accuracy", self._lm_accuracy)])
      self.mono_summary = tf.summary.merge([
          self._base_summaries,
          tf.summary.scalar("semi_supervised_tm_accuracy", self._tm_accuracy),
          tf.summary.scalar("semi_supervised_ELBO", self._elbo),
          tf.summary.scalar("semi_supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("semi_supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("semi_supervised_KL_Z", self._KL_Z),
          tf.summary.scalar("semi_supervised_entropy", self._entropy)])

  # Overrides model.build_graph
  def build_graph(self, hparams, scope=None):
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      # Infer z from the embeddings
      Z = self._infer_z_from_embeddings(hparams)
      z_sample = Z.sample()

      with tf.variable_scope("generative_model", dtype=dtype):

        # P(x_1^m) language model
        lm_logits = self._build_language_model(hparams, z_sample=z_sample)

        # P(y_1^n|x_1^m) encoder
        encoder_outputs, encoder_state = self._build_encoder(hparams,
            z_sample=z_sample)

        # P(y_1^n|x_1^m) decoder
        tm_logits, sample_id, final_context_state = self._build_decoder(
            encoder_outputs, encoder_state, hparams, z_sample=z_sample)

        # Loss
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
          with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                     self.num_gpus)):
            loss, components = self._compute_loss(tm_logits, lm_logits, Z)
        else:
          loss = None

    # Save for summaries.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self._tm_loss = components[0]
      self._lm_loss = components[1]
      self._KL_Z = components[2]
      self._entropy = components[3]
      self._elbo = -loss

      self._lm_accuracy = self._compute_accuracy(lm_logits,
          tf.argmax(self.source_output, axis=-1, output_type=tf.int32),
          self.source_sequence_length)

    return tm_logits, loss, final_context_state, sample_id

  def _infer_z_from_embeddings(self, hparams):
    with tf.variable_scope("z_inference_model") as scope:
      dtype = scope.dtype
      num_layers = self.num_encoder_layers
      num_residual_layers = self.num_encoder_residual_layers
      num_bi_layers = int(num_layers / 2)
      num_bi_residual_layers = int(num_residual_layers / 2)

      # Use the generative embeddings but don't allow gradients to flow there.
      embeddings = tf.stop_gradient(self._source_embedding(self.source))
      if self.time_major:
        embeddings = self._transpose_time_major(embeddings)

      encoder_outputs, _ = (
          self._build_bidirectional_rnn(inputs=embeddings,
                                        sequence_length=self.source_sequence_length,
                                        dtype=dtype,
                                        hparams=hparams,
                                        num_bi_layers=num_bi_layers,
                                        num_bi_residual_layers=num_bi_residual_layers)
                            )

      # Average the transformed encoder outputs over the time dimension to
      # get a single vector as input to the inference network for z.
      # average_encoding: [batch, num_units]
      max_source_time = self.get_max_time(encoder_outputs)
      mask = tf.sequence_mask(self.source_sequence_length,
          dtype=encoder_outputs.dtype, maxlen=max_source_time)
      if self.time_major: mask = tf.transpose(mask)
      mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, 2*hparams.num_units])
      time_axis = 0 if self.time_major else 1
      average_encoding = tf.reduce_mean(mask * encoder_outputs,
          axis=time_axis)

      # Use the averaged encoding to predict mu and sigma^2 in separate FFNNs.
      with tf.variable_scope("mean_inference_network"):
        z_mu = tf.layers.dense(
            tf.layers.dense(average_encoding, hparams.z_dim,
                activation=tf.nn.relu),
            hparams.z_dim,
            activation=None)

      with tf.variable_scope("stddev_inference_network"):
        z_sigma = tf.layers.dense(
            tf.layers.dense(average_encoding, hparams.z_dim,
                activation=tf.nn.relu),
            hparams.z_dim,
            activation=tf.nn.softplus)

    return tf.contrib.distributions.MultivariateNormalDiag(z_mu, z_sigma)

  def _infer_z_from_encodings(self, encoder_outputs, hparams):

    with tf.variable_scope("z_inference_model"):

      # Make sure no gradients from the inference network flow back to the
      # generative part of the model.
      encoder_outputs = tf.stop_gradient(encoder_outputs)

      # Transform the generative encoder outputs with a single-layer FFNN.
      # transformed_outputs: [batch/time, time/batch, num_units]
      transformed_outputs = tf.layers.dense(
          tf.layers.dense(encoder_outputs, hparams.num_units,
              activation=tf.nn.relu),
          hparams.num_units,
          activation=None,
          name="input_transform")

      # Average the transformed encoder outputs over the time dimension to
      # get a single vector as input to the inference network for z.
      # average_encoding: [batch, num_units]
      max_source_time = self.get_max_time(encoder_outputs)
      mask = tf.sequence_mask(self.source_sequence_length,
          dtype=transformed_outputs.dtype, maxlen=max_source_time)
      if self.time_major: mask = tf.transpose(mask)
      mask = tf.tile(tf.expand_dims(mask, axis=-1), [1, 1, hparams.num_units])
      time_axis = 0 if self.time_major else 1
      average_encoding = tf.reduce_mean(mask * transformed_outputs,
          axis=time_axis)

      # Use the averaged encoding to predict mu and sigma^2 in separate FFNNs.
      with tf.variable_scope("mean_inference_network"):
        z_mu = tf.layers.dense(
            tf.layers.dense(average_encoding, hparams.z_dim,
                activation=tf.nn.relu),
            hparams.z_dim,
            activation=None)

      with tf.variable_scope("stddev_inference_network"):
        z_sigma = tf.layers.dense(
            tf.layers.dense(average_encoding, hparams.z_dim,
                activation=tf.nn.relu),
            hparams.z_dim,
            activation=tf.nn.softplus)

    return tf.contrib.distributions.MultivariateNormalDiag(z_mu, z_sigma)

  # Overrides SimpleJointModel._compute_loss
  def _compute_loss(self, tm_logits, lm_logits, Z):

    # The cross-entropy under a reparameterizable sample of the latent variable(s).
    tm_loss = self._compute_categorical_loss(tm_logits,
        self.target_output, self.target_sequence_length)

    # The cross-entropy for the language model also under a sample of the latent
    # variable(s). Not correct mathematically, if we use the relaxation.
    lm_loss = self._compute_dense_categorical_loss(lm_logits,
        self.source_output, self.source_sequence_length)

    # We use a heuristic as an unjustified approximation for monolingual
    # batches.
    max_source_time = self.get_max_time(lm_logits)
    source_weights = tf.sequence_mask(self.source_sequence_length,
        max_source_time, dtype=lm_logits.dtype)
    entropy = tf.cond(self.mono_batch,
        true_fn=lambda: self._compute_categorical_entropy(self.source,
                                                          source_weights),
        false_fn=lambda: tf.constant(0.))

    # We compute an analytical KL between the Gaussian variational approximation
    # and its Gaussian prior.
    standard_normal = tf.contrib.distributions.MultivariateNormalDiag(
        tf.zeros_like(Z.mean()), tf.ones_like(Z.stddev()))
    KL_Z = Z.kl_divergence(standard_normal)
    KL_Z = tf.reduce_mean(KL_Z)

    return tm_loss + lm_loss + KL_Z - entropy, (tm_loss, lm_loss, KL_Z, entropy)
