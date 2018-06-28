"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt.baseline import BaselineModel
from nmt import model_helper
from nmt.utils.amt_utils import enrich_embeddings_with_positions, self_attention_layer, diagonal_attention_coefficients
from nmt.contrib.stat.dist import Gumbel
from nmt.utils.gumbelhelper import GumbelHelper
from nmt.joint.utils import make_initial_state, language_model

class DSimpleJointModel(BaselineModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None, no_summaries=False):

    # Currently here just for consistency because Q(X) uses GRU cells.
    assert hparams.unit_type == "gru"

    self.gumbel = Gumbel()

    super(DSimpleJointModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)

    self.supports_monolingual = True

    # Set model specific training summaries.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN and not no_summaries:
      self.bi_summary = tf.summary.merge([
          self._base_summaries,
          self._supervised_tm_accuracy_summary,
          tf.summary.scalar("supervised_ELBO", self._elbo),
          tf.summary.scalar("supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("supervised_lm_accuracy", self._lm_accuracy)])
      self.mono_summary = tf.summary.merge([
          self._base_summaries,
          tf.summary.scalar("semi_supervised_tm_accuracy", self._tm_accuracy),
          tf.summary.scalar("semi_supervised_ELBO", self._elbo),
          tf.summary.scalar("semi_supervised_tm_loss", self._tm_loss),
          tf.summary.scalar("semi_supervised_lm_loss", self._lm_loss),
          tf.summary.scalar("semi_supervised_entropy", self._entropy)])


  # Overrides Model._parse_iterator
  # Parses the data iterator and sets instance variables correctly.
  def _parse_iterator(self, iterator, hparams, scope=None):
    dtype = tf.float32
    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):

      self.initializer = iterator.initializer
      self.mono_initializer = iterator.mono_initializer
      self.mono_batch = iterator.mono_batch

      # No semi-supervised training for back-translation data.
      if hparams.synthetic_prefix:
        self.mono_batch = tf.constant(False)

      # Change the data depending on what type of batch we're training on.
      self.target_input, self.target_output, self.target_sequence_length = tf.cond(
          self.mono_batch,
          true_fn=lambda: (iterator.mono_text_input, iterator.mono_text_output,
                           iterator.mono_text_length),
          false_fn=lambda: (iterator.target_input, iterator.target_output,
                            iterator.target_sequence_length))

      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        self.batch_size = tf.size(self.target_sequence_length)
      else:
        self.batch_size = tf.size(iterator.source_sequence_length)

      self.source, self.source_output, self.source_sequence_length = tf.cond(
          self.mono_batch,
          true_fn=lambda: self._infer_source(iterator, hparams),
          false_fn=lambda: (tf.one_hot(iterator.source, self.src_vocab_size,
                                       dtype=tf.float32),
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
            loss, components = self._compute_loss(tm_logits, lm_logits)
        else:
          loss = None

    # Save for summaries.
    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
      self._tm_loss = components[0]
      self._lm_loss = components[1]
      self._entropy = components[2]
      self._elbo = -loss

      self._lm_accuracy = self._compute_accuracy(lm_logits,
          tf.argmax(self.source_output, axis=-1, output_type=tf.int32),
          self.source_sequence_length)

    return tm_logits, loss, final_context_state, sample_id

  # Overrides BaselineModel._source_embedding
  def _source_embedding(self, source):
    return tf.tensordot(source, self.embedding_encoder, axes=[[2], [0]])

  # Builds a Categorical language model. If z_sample is given it will be used
  # to initialize the RNNLM.
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

    # Put the RNN output through a projection layer to obtain the logits.
    logits = tf.layers.dense(
        lm_outputs.rnn_output,
        self.src_vocab_size,
        name="output_projection")

    return logits

  def _positional_encoder(self, target, target_length, hparams):

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

  def _birnn_encoder(self, target, target_length, hparams):

    # [batch, time, num_units]
    embeddings = tf.nn.embedding_lookup(
        tf.stop_gradient(self.embedding_decoder), target)

    if self.time_major:
      embeddings = self._transpose_time_major(embeddings)

    fw_cell = tf.contrib.rnn.GRUCell(hparams.num_units)
    bw_cell = tf.contrib.rnn.GRUCell(hparams.num_units)

    encoder_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        embeddings,
        sequence_length=target_length,
        time_major=self.time_major,
        dtype=embeddings.dtype)

    final_state = tf.concat(final_state, axis=-1)
    encoder_outputs = tf.concat(encoder_outputs, axis=-1)

    if self.time_major:
      encoder_outputs = self._transpose_time_major(encoder_outputs)

    return encoder_outputs, final_state

  def _sutskever_encoder(self, target, target_length, hparams):

    reverse_target = tf.reverse(target, axis=[-1])

    # [batch, time, num_units]
    embeddings = tf.nn.embedding_lookup(
        tf.stop_gradient(self.embedding_decoder), reverse_target)

    if self.time_major:
      embeddings = self._transpose_time_major(embeddings)

    cell = tf.contrib.rnn.GRUCell(hparams.num_units)

    encoder_outputs, final_state = tf.nn.dynamic_rnn(cell, embeddings,
        sequence_length=target_length,
        time_major=self.time_major,
        dtype=embeddings.dtype)

    if self.time_major:
      encoder_outputs = self._transpose_time_major(encoder_outputs)

    return encoder_outputs, final_state

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
    logits = tf.layers.dense(attention_output, self.src_vocab_size)
    std_gumbel_sample = self.gumbel.random_standard(tf.shape(logits))
    inferred_source = tf.nn.softmax(logits + std_gumbel_sample)

    return inferred_source

  def _deterministic_rnn_decoder(self, encoder_outputs, final_state,
      target_length, predicted_source_length, hparams):

    max_source_length = tf.reduce_max(predicted_source_length)
    inputs = tf.tile(tf.expand_dims(final_state, 1),
        [1, max_source_length, 1])
    inputs = enrich_embeddings_with_positions(inputs,
        hparams.num_units, "positional_embeddings")
    if self.time_major:
      inputs = self._transpose_time_major(inputs)

    cell = tf.contrib.rnn.GRUCell(hparams.num_units)
    decoder_outputs, _ = tf.nn.dynamic_rnn(cell, inputs,
        sequence_length=predicted_source_length,
        time_major=self.time_major,
        dtype=inputs.dtype)

    # Return batch major.
    if self.time_major:
      decoder_outputs = self._transpose_time_major(decoder_outputs)

    logits = tf.layers.dense(decoder_outputs, self.src_vocab_size)
    std_gumbel_sample = self.gumbel.random_standard(tf.shape(logits))
    inferred_source = tf.nn.softmax(logits + std_gumbel_sample)

    return inferred_source

  def _rnn_decoder(self, encoder_outputs, encoder_state, target_length,
                   predicted_source_length, hparams):
    scope = tf.get_variable_scope()
    if self.time_major:
      encoder_outputs = self._transpose_time_major(encoder_outputs)

    # Create an identical cell to the forward NMT decoder, but disable
    # inference mode.
    cell, decoder_init_state = self._build_decoder_cell(hparams,
        encoder_outputs, encoder_state, target_length, no_infer=True)

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
    projection_layer = tf.layers.Dense(hparams.src_vocab_size)
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

  # Infers the source sentence from target data. If embeddings is True,
  # will assume we are inferring embeddings instead of word categories.
  def _infer_source(self, iterator, hparams, embeddings=False):
    with tf.variable_scope("source_inference_model"):
      predicted_source_length = iterator.predicted_source_length
      target = iterator.mono_text_input
      target_length = iterator.mono_text_length

      # Limit the length of the source sentences.
      max_length = tf.fill(tf.shape(predicted_source_length), hparams.src_max_len)
      predicted_source_length = tf.minimum(predicted_source_length, max_length)

      # Encode the target sentence.
      with tf.variable_scope("encoder") as scope:
        if hparams.Qx_encoder == "positional":
          encoder_outputs = self._positional_encoder(target, target_length,
              hparams)
          encoder_state = None
        elif hparams.Qx_encoder == "birnn":
          encoder_outputs, encoder_state = self._birnn_encoder(target,
              target_length, hparams)
        elif hparams.Qx_encoder == "sutskever":
          encoder_outputs, encoder_state = self._sutskever_encoder(target,
              target_length, hparams)
        else:
          raise ValueError("Unknown Qx_encoder type: %s" % hparams.Qx_encoder)

      # Infer a probabilistic source sentence.
      with tf.variable_scope("decoder"):
        if hparams.Qx_decoder == "diagonal":
          inferred_source = self._diagonal_decoder(encoder_outputs, target_length,
              predicted_source_length, hparams)
        elif hparams.Qx_decoder == "rnn":
          inferred_source = self._rnn_decoder(encoder_outputs, encoder_state,
              target_length, predicted_source_length, hparams)
        elif hparams.Qx_decoder == "det_rnn":
          inferred_source = self._deterministic_rnn_decoder(encoder_outputs,
              encoder_state, target_length, predicted_source_length, hparams)
        else:
          raise ValueError("Unknown Qx_decoder type: %s" % hparams.Qx_decoder)

      # Create <s> tokens.
      src_sos_id = tf.cast(self.src_vocab_table.lookup(
          tf.constant(hparams.sos)), tf.int32)
      start_tokens = tf.fill([self.batch_size], src_sos_id)

      # Depending on if we're dealing with embeddings or word categories,
      # either embed or one_hot.
      if embeddings:
        vectorizing_fn = lambda x: tf.nn.embedding_lookup(
            self.embedding_encoder, x)
      else:
        vectorizing_fn = lambda x: tf.one_hot(x, hparams.src_vocab_size,
                                              dtype=tf.float32)

      # Now create an input and an output version for the LM, with <s>
      # appended to the beginning for the input, and the extra predicted
      # symbol at the end for the output.
      time_axis = 1
      start_tokens = tf.expand_dims(
          vectorizing_fn(start_tokens),
          axis=time_axis)

      source = tf.concat((start_tokens, inferred_source), time_axis)

      # Mask out all tokens outside of predicted source length with end-of-sentence
      # one-hot vectors.
      inferred_source = tf.concat((inferred_source, start_tokens), time_axis)
      src_eos_id = tf.cast(self.src_vocab_table.lookup(tf.constant(hparams.eos)),
          tf.int32)
      eos_matrix = vectorizing_fn(tf.fill(tf.shape(inferred_source)[:-1],
          src_eos_id))
      max_seq_len = tf.reduce_max(predicted_source_length) + 1
      multiplier = self.src_vocab_size if not embeddings else self.src_embed_size
      seq_mask = tf.tile(tf.expand_dims(tf.sequence_mask(predicted_source_length,
          dtype=tf.bool, maxlen=max_seq_len), axis=-1),
          multiples=[1, 1, multiplier])
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

  # Computes the entropy of a sequence of categorical variables
  # as the sum of their individual entropies.
  def _compute_categorical_entropy(self, probabilities, sequence_mask):
    entropy = -tf.reduce_sum(probabilities * tf.log(probabilities + self.epsilon),
        axis=-1)
    entropy = tf.reduce_sum(sequence_mask * entropy) / tf.to_float(self.batch_size)
    return entropy

  # A mathematically unjustified heuristic that assumes X is Categorical,
  # even though it is a dense Concrete variable. Splits up the KL in
  # a categorical cross-entropy and a categorical entropy.
  def _KL_heuristic(self, lm_logits):
    lm_loss = self._compute_dense_categorical_loss(lm_logits,
        self.source_output, self.source_sequence_length)
    max_source_time = self.get_max_time(lm_logits)
    source_weights = tf.sequence_mask(self.source_sequence_length,
        max_source_time, dtype=lm_logits.dtype)
    entropy = self._compute_categorical_entropy(self.source, source_weights)
    return lm_loss - entropy

  # Overrides Model._compute_loss
  def _compute_loss(self, tm_logits, lm_logits):
    tm_loss = self._compute_categorical_loss(tm_logits,
        self.target_output, self.target_sequence_length)

    # This is mathematically unjustified, but acts together with the entropy
    # as a heuristic to compute the infeasible loss.
    lm_loss = self._compute_dense_categorical_loss(lm_logits,
        self.source_output, self.source_sequence_length)

    max_source_time = self.get_max_time(lm_logits)
    source_weights = tf.sequence_mask(self.source_sequence_length,
        max_source_time, dtype=lm_logits.dtype)
    entropy = tf.cond(self.mono_batch,
        true_fn=lambda: self._compute_categorical_entropy(self.source,
                                                          source_weights),
        false_fn=lambda: tf.constant(0.))

    return tm_loss + lm_loss - entropy, (tm_loss, lm_loss, entropy)
