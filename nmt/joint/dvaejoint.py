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

    # Create the complexity factor for the KL. This way its value can be
    # overriden in a feed dict during runtime.
    self.complexity_factor = tf.constant(1.0)

    super(DVAEJointModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args, no_summaries=True)

    self.has_KL = True

    # Set model specific training summaries.
    complexity_factor_summary = tf.summary.scalar("complexity_factor",
        self.complexity_factor)
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN and not no_summaries:
      self.bi_summary = tf.summary.merge([
          self._base_summaries,
          complexity_factor_summary,
          self._supervised_tm_accuracy_summary,
          tf.summary.scalar("supervised_ELBO", self._elbo),
          tf.summary.scalar("supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("supervised_KL_Z", self._KL_Z),
          tf.summary.scalar("supervised_KL_Z_networks", self._KL_Z_networks),
          tf.summary.scalar("supervised_lm_accuracy", self._lm_accuracy)])
      self.mono_summary = tf.summary.merge([
          self._base_summaries,
          complexity_factor_summary,
          tf.summary.scalar("semi_supervised_tm_accuracy", self._tm_accuracy),
          tf.summary.scalar("semi_supervised_ELBO", self._elbo),
          tf.summary.scalar("semi_supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("semi_supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("semi_supervised_KL_Z", self._KL_Z),
          tf.summary.scalar("supervised_KL_Z_networks", self._KL_Z_networks),
          tf.summary.scalar("semi_supervised_entropy", self._entropy)])

  # Overrides Model.eval
  def eval(self, sess):
    assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    return sess.run([self.eval_loss,
        self.KL, self.predict_count, self.batch_size])

  # Infers z from embeddings, using either fully or less amortized VI.
  # Returns a sample (or the mean), and the latent variables themselves.
  def infer_z(self, hparams):

    # Infer z from the embeddings
    if hparams.z_inference_from == "source_only":
      utils.print_out(" Inferring z from source only")
      Z_x = self._infer_z_from_embeddings(hparams, use_target=False)

      # Either use a sample or the mean.
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        z_sample = Z_x.sample()
      else:
        z_sample = Z_x.mean()

      return z_sample, Z_x
    elif hparams.z_inference_from == "source_target":
      utils.print_out(" Inferring z from both source and target")
      Z_xy = self._infer_z_from_embeddings(hparams,
          scope_name="z_inference_model_xy", use_target=True)
      Z_x = self._infer_z_from_embeddings(hparams,
          scope_name="z_inference_model_x", use_target=False)

      # Either use a sample or the mean.
      if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
        z_sample = Z_xy.sample()
      else:
        z_sample = Z_x.mean()

      return z_sample, (Z_x, Z_xy)
    else:
      raise ValueError("Unknown z inference from option:"
          " %s" % hparams.z_inference_from)

  # Overrides model.build_graph
  def build_graph(self, hparams, scope=None):
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      z_sample, Z = self.infer_z(hparams)

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

            loss, components = self._compute_loss(tm_logits, lm_logits,
                Z, Z_source_target=(hparams.z_inference_from == "source_target"))
        else:
          loss = None

    # Save for summaries.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self._tm_loss = components[0]
      self._lm_loss = components[1]
      self._KL_Z = components[2]
      self._entropy = components[3]
      self._KL_Z_networks = components[4]
      self._elbo = -loss

      self._lm_accuracy = self._compute_accuracy(lm_logits,
          tf.argmax(self.source_output, axis=-1, output_type=tf.int32),
          self.source_sequence_length)

    return tm_logits, loss, final_context_state, sample_id

  def _infer_z_from_embeddings(self, hparams, scope_name="z_inference_model",
      use_target=False):
    with tf.variable_scope(scope_name) as scope:
      dtype = scope.dtype
      num_layers = self.num_encoder_layers
      num_residual_layers = self.num_encoder_residual_layers
      num_bi_layers = int(num_layers / 2)
      num_bi_residual_layers = int(num_residual_layers / 2)

      # Use the generative embeddings but don't allow gradients to flow there.
      embeddings = tf.stop_gradient(self._source_embedding(self.source))
      if self.time_major:
        embeddings = self._transpose_time_major(embeddings)

      with tf.variable_scope("source_sentence_encoder") as scope:
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

      # If set, also use the encoded target sequence for inferring z.
      if use_target:
        with tf.variable_scope("target_sentence_encoder") as scope:
          tgt_embeddings = tf.stop_gradient(
              tf.nn.embedding_lookup(self.embedding_decoder, self.target_input))
          if self.time_major:
            tgt_embeddings = self._transpose_time_major(tgt_embeddings)

          tgt_encoder_outputs, _ = (
              self._build_bidirectional_rnn(inputs=tgt_embeddings,
                                            sequence_length=self.target_sequence_length,
                                            dtype=dtype,
                                            hparams=hparams,
                                            num_bi_layers=num_bi_layers,
                                            num_bi_residual_layers=num_bi_residual_layers)
                                )

          # Average the transformed encoder outputs over the time dimension to
          # get a single vector as input to the inference network for z.
          # average_encoding: [batch, num_units]
          max_target_time = self.get_max_time(tgt_encoder_outputs)
          tgt_mask = tf.sequence_mask(self.target_sequence_length,
              dtype=tgt_encoder_outputs.dtype, maxlen=max_target_time)
          if self.time_major: tgt_mask = tf.transpose(tgt_mask)
          tgt_mask = tf.tile(tf.expand_dims(tgt_mask, axis=-1), [1, 1, 2*hparams.num_units])
          time_axis = 0 if self.time_major else 1
          average_tgt_encoding = tf.reduce_mean(tgt_mask * tgt_encoder_outputs,
              axis=time_axis)

        # Concatenate the source and target average encoders.
        average_encoding = tf.concat([average_encoding, average_tgt_encoding],
            axis=-1)

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
  def _compute_loss(self, tm_logits, lm_logits, Z, Z_source_target=False):

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
    if Z_source_target:
      Z_x, Z_xy = Z
      if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
        KL_Z_networks = tf.constant(0.)
      else:
        Z_xy_sg = tf.contrib.distributions.MultivariateNormalDiag(
            tf.stop_gradient(Z_xy.mean()), tf.stop_gradient(Z_xy.stddev()))
        KL_Z_networks = Z_xy_sg.kl_divergence(Z_x) + Z_x.kl_divergence(Z_xy_sg)
        KL_Z_networks = tf.reduce_mean(KL_Z_networks)

      standard_normal = tf.contrib.distributions.MultivariateNormalDiag(
          tf.zeros_like(Z_xy.mean()), tf.ones_like(Z_xy.stddev()))
      KL_Z = Z_xy.kl_divergence(standard_normal)
    else:
      standard_normal = tf.contrib.distributions.MultivariateNormalDiag(
          tf.zeros_like(Z.mean()), tf.ones_like(Z.stddev()))
      KL_Z = Z.kl_divergence(standard_normal)
      KL_Z_networks = tf.constant(0.)

    KL_Z = tf.reduce_mean(KL_Z)
    self.KL = KL_Z

    return tm_loss + lm_loss + self.complexity_factor * KL_Z - entropy + KL_Z_networks, \
        (tm_loss, lm_loss, KL_Z, entropy, KL_Z_networks)
