"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt.joint.utils import make_initial_state
from nmt import model_helper
from .baseline import BaselineModel

class CBaselineModel(BaselineModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None):

    super(CBaselineModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)

  # Overrides Model._parse_iterator
  # Returns word embeddings instead of one hot vectors.
  def _parse_iterator(self, iterator, hparams, scope=None):
    dtype = tf.float32
    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
      self.src_embed_size = self.embedding_encoder.shape[1]
      self.initializer = iterator.initializer
      self.mono_initializer = iterator.mono_initializer
      self.mono_batch = iterator.mono_batch

      # Change the data depending on what type of batch we're training on.
      self.target_input, self.target_output, self.target_sequence_length = (iterator.target_input, iterator.target_output, iterator.target_sequence_length)

      self.batch_size = tf.size(self.target_sequence_length)
      self.source, self.source_output, self.source_sequence_length = (tf.nn.embedding_lookup(self.embedding_encoder,
                                                  iterator.source),
                            tf.nn.embedding_lookup(self.embedding_encoder,
                                                   iterator.source_output),
                            iterator.source_sequence_length)

  # Overrides Model._source_embedding
  # We use pre-trained embeddings, thus don't do an embedding lookup.
  def _source_embedding(self, source):
    return source
