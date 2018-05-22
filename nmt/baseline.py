"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

from nmt.attention_model import AttentionModel

class BaselineModel(AttentionModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None):

    # Make sure some requirements on the hyperparameters are met.
    assert hparams.unit_type == "lstm"
    assert hparams.encoder_type == "bi"
    assert hparams.num_encoder_layers == 2

    super(BaselineModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)
