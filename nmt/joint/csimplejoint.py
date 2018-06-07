"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from . import DSimpleJointModel
from nmt import model_helper
from .utils import language_model
from nmt.utils.gaussianhelper import GaussianHelper
from nmt.utils.amt_utils import enrich_embeddings_with_positions, self_attention_layer, diagonal_attention_coefficients

class CSimpleJointModel(DSimpleJointModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None, no_summaries=False):

    assert hparams.src_embed_file

    super(CSimpleJointModel, self).__init__(hparams=hparams, mode=mode,
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
          tf.summary.scalar("supervised_lm_loss", self._lm_loss)])
      self.mono_summary = tf.summary.merge([
          self._base_summaries,
          tf.summary.scalar("semi_supervised_tm_accuracy", self._tm_accuracy),
          tf.summary.scalar("semi_supervised_ELBO", self._elbo),
          tf.summary.scalar("semi_supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("semi_supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("semi_supervised_entropy", self._entropy)])

  # Overrides DSimpleJointModel._source_embedding
  # We use pre-trained embeddings, thus don't do an embedding lookup.
  def _source_embedding(self, source):
    return source

  # Overrides DSimpleJointModel._parse_iterator
  # Returns word embeddings instead of one hot vectors.
  def _parse_iterator(self, iterator, hparams, scope=None):
    dtype = tf.float32
    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
      self.src_embed_size = self.embedding_encoder.shape[1]
      self.initializer = iterator.initializer
      self.mono_initializer = iterator.mono_initializer
      self.mono_batch = iterator.mono_batch

      # Change the data depending on what type of batch we're training on.
      self.target_input, self.target_output, self.target_sequence_length = tf.cond(
          self.mono_batch,
          true_fn=lambda: (iterator.mono_text_input, iterator.mono_text_output,
                   iterator.mono_text_length),
          false_fn=lambda: (iterator.target_input, iterator.target_output,
                   iterator.target_sequence_length))
      self.batch_size = tf.size(self.target_sequence_length)

      self.source, self.source_output, self.source_sequence_length = tf.cond(
          self.mono_batch,
          true_fn=lambda: self._infer_source(iterator, hparams, embeddings=True),
          false_fn=lambda: (tf.nn.embedding_lookup(self.embedding_encoder,
                                                  iterator.source),
                            tf.nn.embedding_lookup(self.embedding_encoder,
                                                   iterator.source_output),
                            iterator.source_sequence_length))

  # Builds a Gaussian language model with fixed diagonal unit variance.
  # If z_sample is given it will be used to initialize the RNNLM.
  def _build_language_model(self, hparams, z_sample=None):
    source = self.source
    if self.time_major:
      source = self._transpose_time_major(source)

    # Use embeddings as inputs.
    embeddings = self._source_embedding(source)

    # Run the RNNLM.
    lm_outputs = language_model(embeddings, self.source_sequence_length,
        hparams, self.mode, self.single_cell_fn, self.time_major,
        self.batch_size, z_sample=z_sample)

    # Put the RNN output through a projection layer to obtain a mean for the
    # Gaussians.
    mean = tf.layers.dense(
        lm_outputs.rnn_output,
        self.src_embed_size,
        name="output_projection")

    stddev = tf.ones_like(mean)

    return tf.contrib.distributions.MultivariateNormalDiag(loc=mean,
        scale_diag=stddev)

  # Overrides DSimpleJointModel.build_graph
  def build_graph(self, hparams, scope=None):
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      with tf.variable_scope("generative_model", dtype=dtype):

        # P(x_1^m) language model
        gauss_observations = self._build_language_model(hparams)

        # P(y_1^n|x_1^m) encoder
        encoder_outputs, encoder_state = self._build_encoder(hparams)

        # P(y_1^n|x_1^m) decoder
        tm_logits, sample_id, final_context_state = self._build_decoder(
            encoder_outputs, encoder_state, hparams)

        # Loss
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
          with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                     self.num_gpus)):
            loss, components = self._compute_loss(tm_logits, gauss_observations)
        else:
          loss = None

    # Save for summaries.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self._tm_loss = components[0]
      self._lm_loss = components[1]
      self._entropy = components[2]
      self._elbo = -loss

    return tm_logits, loss, final_context_state, sample_id

  # Overrides DSimpleJointModel._diagonal_decoder
  # Predicts Gaussian X_i | x_<i, y_1^n.
  def _diagonal_decoder(self, encoder_outputs, target_length,
                        predicted_source_length, hparams):

      # Tile encoder_outputs from [B x T_i x d] to [B x T_o x T_i x d]
      encoder_outputs = tf.expand_dims(encoder_outputs, axis=1)
      encoder_outputs = tf.tile(encoder_outputs,
          multiples=[1, tf.reduce_max(predicted_source_length), 1, 1])

      # Create source and target sequence masks.
      y_mask = tf.sequence_mask(target_length, dtype=tf.float32)
      x_mask = tf.sequence_mask(predicted_source_length,
          dtype=tf.float32)

      # Compute fixed decoder coefficients based only on the source and
      # target sentence length.
      attention_coefficients = diagonal_attention_coefficients(y_mask, x_mask,
          target_length, predicted_source_length)
      attention_coefficients = tf.expand_dims(attention_coefficients, axis=-1)
      attention_output = tf.reduce_sum(encoder_outputs * attention_coefficients,
          axis=2)

      # Use the attention output to predict a mean and a diagonal covariance
      # for X_i.
      mean = tf.layers.dense(attention_output, self.src_embed_size)
      stddev = tf.layers.dense(attention_output, self.src_embed_size)

      self.Qx = tf.contrib.distributions.MultivariateNormalDiag(loc=mean,
          scale_diag=stddev)

      return self.Qx.sample()

  # Overrides DSimpleJointModel._rnn_decoder
  # Models X_i | y_1^n, x_<i as Gaussian variables.
  def _rnn_decoder(self, encoder_outputs, encoder_state, target_length,
                   predicted_source_length, hparams):
    scope = tf.get_variable_scope()
    if self.time_major:
      encoder_outputs = self._transpose_time_major(encoder_outputs)

    # Create an identical cell to the forward NMT decoder, but disable
    # inference mode.
    cell, decoder_init_state = self._build_decoder_cell(hparams,
        encoder_outputs, encoder_state, target_length, no_infer=True)

    # Create the initial inputs for the decoder. Use the generative embedding
    # matrix but stop the gradients.
    src_sos_id = tf.cast(self.src_vocab_table.lookup(
        tf.constant(hparams.sos)), tf.int32)
    start_tokens = tf.fill([self.batch_size], src_sos_id)
    start_tokens = tf.nn.embedding_lookup(self.embedding_encoder, start_tokens)
    start_tokens = tf.stop_gradient(start_tokens)

    # Create the Gaussian helper to generate Gaussian samples.
    helper = GaussianHelper(
        start_tokens=start_tokens,
        decode_lengths=predicted_source_length)
    utils.print_out("  creating GaussianHelper")

    # Create the decoder.
    projection_layer = tf.layers.Dense(self.src_embed_size * 2)
    decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper,
        decoder_init_state, output_layer=projection_layer)

    # Decode the Concrete source sentence.
    outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=self.time_major,
        maximum_iterations=tf.reduce_max(predicted_source_length),
        swap_memory=True,
        scope=scope)
    inferred_source = outputs.sample_id

    mean = outputs.rnn_output[:, :, :self.src_embed_size]
    stddev = outputs.rnn_output[:, :, self.src_embed_size:]

    # Return in batch major.
    if self.time_major:
      inferred_source = self._transpose_time_major(inferred_source)
      mean = self._transpose_time_major(mean)
      stddev = self._transpose_time_major(stddev)

    self.Qx = tf.contrib.distributions.MultivariateNormalDiag(loc=mean,
        scale_diag=stddev)

    return inferred_source

  # Gives the negative log-likelihood of given observations for a gaussian
  # variable.
  def _gaussian_nll(self, gauss_var, observations, observation_length):
    if self.time_major: observations = self._transpose_time_major(observations)
    log_prob = gauss_var.log_prob(observations)
    max_source_time = self.get_max_time(observations)
    mask = tf.sequence_mask(observation_length, max_source_time,
        dtype=log_prob.dtype)
    if self.time_major: mask = tf.transpose(mask)
    nll = -tf.reduce_sum(log_prob * mask) / tf.to_float(self.batch_size)
    return nll

  # Overrides DSimpleJointModel._compute_loss
  def _compute_loss(self, tm_logits, gauss_observations):

    # - log P(y_1^n)
    tm_loss = self._compute_categorical_loss(tm_logits,
        self.target_output, self.target_sequence_length)

    # - log p(x_1^m)
    lm_loss = self._gaussian_nll(gauss_observations, self.source_output,
        self.source_sequence_length)

    # H(X|y_1^n) -- keep in mind self.Qx is defined in batch major, as are all
    # data streams.
    entropy = tf.cond(self.mono_batch,
        true_fn=lambda: tf.reduce_mean(tf.reduce_sum(self.Qx.entropy(), axis=1)),
        false_fn=lambda: tf.constant(0.))

    return tm_loss + lm_loss - entropy, (tm_loss, lm_loss, entropy)
