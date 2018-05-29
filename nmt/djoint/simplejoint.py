"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt.baseline import BaselineModel
from nmt import model_helper
from nmt.utils.joint_utils import enrich_embeddings_with_positions, self_attention_layer, diagonal_attention_coefficients
from nmt.contrib.stat.dist import Gumbel
from nmt.utils.gumbelhelper import GumbelHelper

class SimpleJointModel(BaselineModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None):

    self.gumbel = Gumbel()

    super(SimpleJointModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)

    self.supports_monolingual = True

  # Overrides Model._parse_iterator
  # Parses the data iterator and sets instance variables correctly.
  def _parse_iterator(self, iterator, hparams):
    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      super(SimpleJointModel, self)._parse_iterator(iterator, hparams)
      self.source = tf.one_hot(self.source, self.src_vocab_size, dtype=tf.float32)
      return

    self.initializer = iterator.initializer
    self.mono_initializer = iterator.mono_initializer
    self.mono_batch = iterator.mono_batch

    # Change the data depending on what type of batch we're training on.
    self.target_input, self.target_output, self.target_sequence_length = tf.cond(
        self.mono_batch,
        lambda: (iterator.mono_text_input, iterator.mono_text_output,
                 iterator.mono_text_length),
        lambda: (iterator.target_input, iterator.target_output,
                 iterator.target_sequence_length)) 
    self.batch_size = tf.size(self.target_sequence_length)

    self.source, self.source_output, self.source_sequence_length = tf.cond(
        self.mono_batch,
        lambda: self._infer_source(iterator, hparams),
        lambda: (tf.one_hot(iterator.source, self.src_vocab_size, dtype=tf.float32),
                 tf.one_hot(iterator.source_output, self.src_vocab_size,
                            dtype=tf.float32),
                 iterator.source_sequence_length))

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

  # Overrides BaselineModel._source_embedding
  def _source_embedding(self, source):
    return tf.tensordot(source, self.embedding_encoder, axes=[[2], [0]])

  def _build_language_model(self, hparams):
    source = self.source
    if self.time_major:
      source = self._transpose_time_major(source)

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
      embeddings = self._source_embedding(source)
      init_state = cell.zero_state(self.batch_size, scope.dtype)

      # Run the LSTM language model.
      helper = tf.contrib.seq2seq.TrainingHelper(
          embeddings,
          self.source_sequence_length,
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

  def _positional_encoder(self, target, target_length, predicted_source_length, hparams):
      # Embed the target sentence with the decoder embedding matrix.
      # We use the generative embedding matrix, but stop gradients from
      # flowing through.
      embeddings = tf.nn.embedding_lookup(
          tf.stop_gradient(self.embedding_decoder), target)
      embeddings = enrich_embeddings_with_positions(embeddings,
          hparams.num_units, "positional_embeddings")

      # Compute self attention.
      attention = self_attention_layer(embeddings, target_length,
          hparams.num_units, mask_diagonal=True)
      other = tf.matmul(attention, embeddings)
      att_output = tf.concat([embeddings, other], axis=-1)

      # Put the output vector through an MLP.
      encoder_outputs = tf.layers.dense(
          tf.layers.dense(att_output, hparams.num_units, activation=tf.nn.relu),
          hparams.num_units,
          activation=None)

      return encoder_outputs

  def _birnn_encoder(self, target, target_length, predicted_source_length, hparams):
    scope = tf.get_variable_scope()
    dtype = scope.dtype
    num_layers = self.num_encoder_layers
    num_residual_layers = self.num_encoder_residual_layers
    num_bi_layers = int(num_layers / 2)
    num_bi_residual_layers = int(num_residual_layers / 2)

    if self.time_major:
      target = self._transpose_time_major(target)

    # Embed the target sentence with the decoder embedding matrix.
    # We use the generative embedding matrix, but stop gradients from
    # flowing through.
    embeddings = tf.nn.embedding_lookup(tf.stop_gradient(self.embedding_decoder),
        target)

    encoder_outputs, bi_encoder_state = (
        self._build_bidirectional_rnn(inputs=embeddings,
                                      sequence_length=target_length,
                                      dtype=dtype,
                                      hparams=hparams,
                                      num_bi_layers=num_bi_layers,
                                      num_bi_residual_layers=num_bi_residual_layers))

    if num_bi_layers == 1:
      encoder_state = bi_encoder_state
    else:
      encoder_state = []
      for layer_id in range(num_bi_layers):
        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
      encoder_state = tuple(encoder_state)

    # Return output in batch major.
    if self.time_major:
      encoder_outputs = self._transpose_time_major(encoder_outputs)

    return encoder_outputs, encoder_state

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

      # Project the attention output to the vocabulary size to obtain the
      # Gumbel parameters.
      logits = tf.layers.dense(attention_output, self.src_vocab_size,
          use_bias=True)
      std_gumbel_sample = self.gumbel.random_standard(tf.shape(logits))
      inferred_source = tf.nn.softmax(logits + std_gumbel_sample)

      return inferred_source

  def _rnn_decoder(self, encoder_outputs, encoder_state, target_length,
                   predicted_source_length, hparams):
    scope = tf.get_variable_scope()
    if self.time_major:
      encoder_outputs = self._transpose_time_major(encoder_outputs)

    # Create an identical cell to the forward NMT decoder.
    cell, decoder_init_state = self._build_decoder_cell(hparams,
        encoder_outputs, encoder_state, target_length)

    # Create the initial inputs for the decoder.
    src_sos_id = tf.cast(self.src_vocab_table.lookup(
        tf.constant(hparams.sos)), tf.int32)
    start_tokens = tf.fill([self.batch_size], src_sos_id)

    # Create the Gumbel helper to generate Concrete samples.
    straight_through = False
    helper = GumbelHelper(
        embedding_matrix=tf.stop_gradient(self.embedding_encoder),
        start_tokens=start_tokens,
        decode_lengths=predicted_source_length,
        straight_through=straight_through)
    utils.print_out("  creating GumbelHelper with straight_through=%s" % \
        straight_through)

    # Create the decoder.
    projection_layer = tf.layers.Dense(hparams.src_vocab_size,
        use_bias=False)
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

    # Return in batch major.
    if self.time_major:
      inferred_source = self._transpose_time_major(inferred_source)

    return inferred_source

  # Infers the source sentence from target data.
  def _infer_source(self, iterator, hparams):
    predicted_source_length = iterator.predicted_source_length
    target = iterator.mono_text_input
    target_length = iterator.mono_text_length

    # Limit the length of the source sentences.
    max_length = tf.fill(tf.shape(predicted_source_length), hparams.src_max_len)
    predicted_source_length = tf.minimum(predicted_source_length, max_length)

    # Encode the target sentence.
    with tf.variable_scope("source_inference_encoder") as scope:
      if hparams.Qx_encoder == "positional":
        encoder_outputs = self._positional_encoder(target, target_length,
            predicted_source_length, hparams)
        encoder_state = None
      elif hparams.Qx_encoder == "birnn":
        encoder_outputs, encoder_state = self._birnn_encoder(target,
            target_length, predicted_source_length, hparams)
      else:
        raise ValueError("Unknown Qx_encoder type: %s" % hparams.Qx_encoder)

    # Infer a Gumbel source sentence.
    with tf.variable_scope("source_inference_decoder"):
      if hparams.Qx_decoder == "diagonal":
        inferred_source = self._diagonal_decoder(encoder_outputs, target_length,
            predicted_source_length, hparams)
      elif hparams.Qx_decoder == "rnn":
        inferred_source = self._rnn_decoder(encoder_outputs, encoder_state,
            target_length, predicted_source_length, hparams)
      else:
        raise ValueError("Unknown Qx_decoder type: %s" % hparams.Qx_decoder)

    # Create <s> tokens.
    src_sos_id = tf.cast(self.src_vocab_table.lookup(
        tf.constant(hparams.sos)), tf.int32)
    start_tokens = tf.fill([self.batch_size], src_sos_id)

    # Now create an input and an output version for the LM, with <s>
    # appended to the beginning for the input, and the extra predicted
    # symbol at the end for the output.
    time_axis = 1
    start_tokens = tf.expand_dims(
        tf.one_hot(start_tokens, hparams.src_vocab_size),
        axis=time_axis)

    source = tf.concat((start_tokens, inferred_source), time_axis)

    # Mask out all tokens outside of predicted source length with end-of-sentence
    # one-hot vectors.
    inferred_source = tf.concat((inferred_source, start_tokens), time_axis)
    src_eos_id = tf.cast(self.src_vocab_table.lookup(tf.constant(hparams.eos)),
        tf.int32)
    eos_matrix = tf.one_hot(tf.fill(tf.shape(inferred_source)[:-1], src_eos_id),
        self.src_vocab_size, dtype=tf.float32)
    max_seq_len = tf.reduce_max(predicted_source_length) + 1
    seq_mask = tf.tile(tf.expand_dims(tf.sequence_mask(predicted_source_length,
        dtype=tf.bool, maxlen=max_seq_len), axis=-1),
        multiples=[1, 1, self.src_vocab_size])
    source_output = tf.where(seq_mask, inferred_source, eos_matrix)

    return source, source_output, predicted_source_length+1

  # Computes the loss of a sequence of categorical variables, given observed data.
  def _compute_categorical_loss(self, logits, observations, seq_length):
    if self.time_major:
      observations = self._transpose_time_major(observations)
    max_time = self.get_max_time(observations)

    # Compute the loss of the categorical variables (cross-entropy)
    categorical_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=observations, logits=logits)

    # Mask out beyond the sequence length.
    mask = tf.sequence_mask(seq_length, max_time, dtype=logits.dtype)
    if self.time_major:
      mask = self._transpose_time_major(mask)

    # Average the loss over the batch.
    avg_loss = tf.reduce_sum(
        categorical_loss * mask) / tf.to_float(self.batch_size)
    return avg_loss

  # Computes the loss of a sequence of categorical variables,
  # given dense observed data.
  def _compute_dense_categorical_loss(self, logits, observations, seq_length):
    if self.time_major:
      observations = self._transpose_time_major(observations)
    max_time = self.get_max_time(observations)

    # Compute the loss of the categorical variables (cross-entropy)
    categorical_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=observations, logits=logits)

    # Mask out beyond the sequence length.
    mask = tf.sequence_mask(seq_length, max_time, dtype=logits.dtype)
    if self.time_major:
      mask = self._transpose_time_major(mask)

    # Average the loss over the batch.
    avg_loss = tf.reduce_sum(
        categorical_loss * mask) / tf.to_float(self.batch_size)
    return avg_loss

  # Overrides model._compute_loss
  def _compute_loss(self, tm_logits, lm_logits):
    tm_loss = self._compute_categorical_loss(tm_logits,
        self.target_output, self.target_sequence_length)
    lm_loss = tf.cond(self.mono_batch,
        lambda: tf.constant(0.),
        lambda: self._compute_dense_categorical_loss(lm_logits, self.source_output,
                                                     self.source_sequence_length))
    KL = tf.cond(self.mono_batch,
        lambda: tf.constant(0.), # TODO
        lambda: tf.constant(0.))
    return tm_loss + lm_loss + KL
