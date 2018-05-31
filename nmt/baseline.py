"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt.attention_model import AttentionModel

class BaselineModel(AttentionModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None):

    # Make sure some requirements on the hyperparameters are met.
    assert hparams.unit_type == "lstm"
    assert hparams.encoder_type == "bi"
    assert hparams.num_encoder_layers == 2

    # For use for numerical stability.
    self.epsilon = 1e-10

    super(BaselineModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)

  # Returns the embeddings for a given source batch.
  def _source_embedding(self, source):
    return tf.nn.embedding_lookup(self.embedding_encoder, source)

  # Function that safely tranposes the batch and time axes, independent
  # whether we are using one hot vectors or not.
  def _transpose_time_major(self, tensor):
    if len(tensor.shape) == 3:
      return tf.transpose(tensor, perm=[1, 0, 2])
    else:
      return tf.transpose(tensor)

  # Overrides model._build_encoder
  # A version that allows for variable ways to look up source embeddings with self._source_embedding(source)
  def _build_encoder(self, hparams):
    """Build an encoder."""
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers

    source = self.source
    if self.time_major:
      source = self._transpose_time_major(source)

    with tf.variable_scope("encoder") as scope:
      dtype = scope.dtype
      # Look up embedding, emp_inp: [max_time, batch_size, num_units]
      encoder_emb_inp = self._source_embedding(source)

      # Encoder_outputs: [max_time, batch_size, num_units]
      if hparams.encoder_type == "uni":
        utils.print_out("  num_layers = %d, num_residual_layers=%d" %
                        (num_layers, num_residual_layers))
        cell = self._build_encoder_cell(
            hparams, num_layers, num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=self.source_sequence_length,
            time_major=self.time_major,
            swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=self.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)
    return encoder_outputs, encoder_state
