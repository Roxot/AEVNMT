"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt.baseline import BaselineModel
from nmt import model_helper

class SimpleJointModel(BaselineModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None):

    super(SimpleJointModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)

  def _build_language_model(self, hparams):
    source = self.iterator.source
    if self.time_major:
      source = tf.transpose(source)

    with tf.variable_scope("language_model") as scope:
      # Use decoder cell options.
      cell = model_helper.create_rnn_cell(
          unit_type="lstm",
          num_units=hparams.num_units,
          num_layers=hparams.num_lm_layers,
          num_residual_layers=hparams.num_decoder_residual_layers,
          forget_bias=hparams.forget_bias,
          dropout=hparams.dropout,
          num_gpus=hparams.num_gpus,
          mode=self.mode,
          single_cell_fn=self.single_cell_fn)

      # Use a zero initial state and the embeddings as inputs.
      embeddings = tf.nn.embedding_lookup(self.embedding_encoder, source)
      init_state = cell.zero_state(self.batch_size, scope.dtype)

      # Run the LSTM language model.
      helper = tf.contrib.seq2seq.TrainingHelper(
          embeddings,
          self.iterator.source_sequence_length,
          time_major=self.time_major)
      decoder = tf.contrib.seq2seq.BasicDecoder(
          cell,
          helper,
          initial_state=init_state)
      lm_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
          decoder,
          output_time_major=self.time_major,
          impute_finished=True,
          scope=scope)

      # Put the LSTM output through a projection layer to obtain the logits.
      logits = tf.layers.dense(
          lm_outputs.rnn_output,
          self.src_vocab_size,
          name="output_projection")

    return logits

  # Computes the loss of a sequence of categorical variables, given observed data.
  def _compute_categorical_loss(self, logits, observations, seq_length):
    if self.time_major:
      observations = tf.transpose(observations)
    max_time = self.get_max_time(observations)

    # Compute the loss of the categorical variables (cross-entropy)
    categorical_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=observations, logits=logits)

    # Mask out beyond the sequence length.
    mask = tf.sequence_mask(seq_length, max_time, dtype=logits.dtype)
    if self.time_major:
      mask = tf.transpose(mask)

    # Average the loss over the batch.
    avg_loss = tf.reduce_sum(
        categorical_loss * mask) / tf.to_float(self.batch_size)
    return avg_loss

  # Overrides model._compute_loss
  def _compute_loss(self, tm_logits, lm_logits):
    tm_loss = self._compute_categorical_loss(tm_logits,
        self.iterator.target_output, self.iterator.target_sequence_length)
    lm_loss = self._compute_categorical_loss(lm_logits,
        self.iterator.source_output, self.iterator.source_sequence_length)
    return tm_loss

  # Overrides model.build_graph
  def build_graph(self, hparams, scope=None):
    utils.print_out("# creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      with tf.variable_scope("generative_model", dtype=dtype):

        # P(x_1^m) language model
        lm_logits = self._build_language_model(hparams)

        # P(y_1^n|x_1^m) encoder
        encoder_outputs, encoder_state = self._build_encoder(hparams)

        # P(y_1^n|x_1^m) decoder
        tm_logits, sample_id, final_context_state = self._build_decoder(
            encoder_outputs, encoder_state, hparams)

        # Loss
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
          with tf.device(model_helper.get_device_str(self.num_encoder_layers - 1,
                                                     self.num_gpus)):
            loss = self._compute_loss(tm_logits, lm_logits)
        else:
          loss = None

    return tm_logits, loss, final_context_state, sample_id
