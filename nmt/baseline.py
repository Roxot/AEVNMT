"""
:Authors: - Bryan Eikema
"""

import tensorflow as tf

import nmt.utils.misc_utils as utils

from nmt.joint.utils import make_initial_state
from nmt.attention_model import AttentionModel
from nmt import model_helper

class BaselineModel(AttentionModel):

  def __init__(self, hparams, mode, iterator, source_vocab_table,
               target_vocab_table, reverse_target_vocab_table=None,
               scope=None, extra_args=None):

    # Make sure some requirements on the hyperparameters are met.
    assert hparams.pass_hidden_state == False

    # For use for numerical stability.
    self.epsilon = 1e-10

    super(BaselineModel, self).__init__(hparams=hparams, mode=mode,
        iterator=iterator, source_vocab_table=source_vocab_table,
        target_vocab_table=target_vocab_table,
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope, extra_args=extra_args)

    if self.mode == tf.contrib.learn.ModeKeys.TRAIN:

      # Set some base summaries that all VI models will use.
      tm_accuracy = self._compute_accuracy(self._logits, self.target_output,
          self.target_sequence_length)
      self._base_summaries = tf.summary.merge([
          self._lr_summary,
          ] + self._grad_norm_summary)

      self._supervised_tm_accuracy_summary =  tf.summary.scalar(
          "supervised_tm_accuracy", tm_accuracy),

      # Also add the tm accuracy summary to the baseline's summaries.
      self.train_summary = tf.summary.merge([
          self.train_summary,
          self._supervised_tm_accuracy_summary])

      # Allows for different summaries for monolingual and bilingual batches.
      self.mono_summary = self.train_summary
      self.bi_summary = self.train_summary
      self._tm_accuracy = tm_accuracy

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
  # A version that allows for variable ways to look up source embeddings with
  # self._source_embedding(source), and that allows for an extra encoder input
  # (VAEJointModel).
  def _build_encoder(self, hparams, z_sample=None):
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

        # If a z sample is provided, use it as the initial state of the encoder.
        if z_sample is not None:
          utils.print_out("  initializing generative encoder with tanh(Wz)")
          init_state_val = tf.tanh(tf.layers.dense(z_sample, hparams.num_units))
          init_state = make_initial_state(init_state_val, hparams.unit_type)
        else:
          utils.print_out("  initializing generative encoder with zeros.")
          init_state = cell.zero_state(self.batch_size)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=self.source_sequence_length,
            time_major=self.time_major,
            initial_state=init_state,
            swap_memory=True)
      elif hparams.encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)
        utils.print_out("  num_bi_layers = %d, num_bi_residual_layers=%d" %
                        (num_bi_layers, num_bi_residual_layers))

        # If a z sample is provided, use it as the initial state of the encoder.
        if z_sample is not None:
          init_state_val = tf.tanh(tf.layers.dense(z_sample, hparams.num_units))
          init_state = make_initial_state(init_state_val, hparams.unit_type)
        else:
          init_state = None

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=self.source_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers,
                initial_state=init_state))

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

  # Overrides Model._build_bidirectional_rnn
  # Allows to set an initial state, same initial state used for both
  # the forward and backward dirdctions.
  def _build_bidirectional_rnn(self, inputs, sequence_length,
                               dtype, hparams,
                               num_bi_layers,
                               num_bi_residual_layers,
                               base_gpu=0, initial_state=None):
    """Create and call biddirectional RNN cells.

    Args:
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`.
      base_gpu: The gpu device id to use for the first forward RNN layer. The
        i-th forward RNN layer will use `(base_gpu + i) % num_gpus` as its
        device id. The `base_gpu` for backward RNN cell is `(base_gpu +
        num_bi_layers)`.

    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=base_gpu)
    bw_cell = self._build_encoder_cell(hparams,
                                       num_bi_layers,
                                       num_bi_residual_layers,
                                       base_gpu=(base_gpu + num_bi_layers))

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length,
        initial_state_fw=initial_state,
        initial_state_bw=initial_state,
        time_major=self.time_major,
        swap_memory=True)

    return tf.concat(bi_outputs, -1), bi_state

  # Overrides Model._build_decoder
  # Accepts a z_sample to set initial state of the decoder (VAEJointModel).
  # Will ignore hparams.pass_hidden_state if z_sample is given.
  def _build_decoder(self, encoder_outputs, encoder_state, hparams,
      z_sample=None):
    """Build and run a RNN decoder with a final projection layer.

    Args:
      encoder_outputs: The outputs of encoder for every time step.
      encoder_state: The final state of the encoder.
      hparams: The Hyperparameters configurations.
      z_sample: If set will initialize the decoder with it.

    Returns:
      A tuple of final logits and final decoder state:
        logits: size [time, batch_size, vocab_size] when time_major=True.
    """
    tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.sos)),
                         tf.int32)
    tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(hparams.eos)),
                         tf.int32)

    # maximum_iteration: The maximum decoding steps.
    maximum_iterations = self._get_infer_maximum_iterations(
        hparams, self.source_sequence_length)

    ## Decoder.
    with tf.variable_scope("decoder") as decoder_scope:
      cell, decoder_initial_state = self._build_decoder_cell(
          hparams, encoder_outputs, encoder_state,
          self.source_sequence_length)

      if z_sample is not None:
        dtype = decoder_scope.dtype
        utils.print_out("  overriding decoder_initial_state with tanh(Wz)")
        init_state_val = tf.tanh(tf.layers.dense(z_sample, hparams.num_units))
        init_state = make_initial_state(init_state_val, hparams.unit_type)

        # If we do beam search we need to tile the initial state
        if self.mode == tf.contrib.learn.ModeKeys.INFER and hparams.beam_width > 0:
          init_state = tf.contrib.seq2seq.tile_batch(
              init_state, multiplier=hparams.beam_width)
          batch_size = self.batch_size * hparams.beam_width
        else:
          batch_size = self.batch_size

        decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
            cell_state=init_state)

      ## Train or eval
      if self.mode != tf.contrib.learn.ModeKeys.INFER:
        # decoder_emp_inp: [max_time, batch_size, num_units]
        target_input = self.target_input
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(
            self.embedding_decoder, target_input)

        # Apply word dropout if set.
        if hparams.word_dropout > 0 and \
            (self.mode == tf.contrib.learn.ModeKeys.TRAIN):

          # Drop random words.
          noise_shape = [tf.shape(decoder_emb_inp)[0],
              tf.shape(decoder_emb_inp)[1], 1]
          decoder_emb_inp = tf.nn.dropout(decoder_emb_inp,
              (1.0 - hparams.word_dropout), noise_shape=noise_shape)

        # Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_emb_inp, self.target_sequence_length,
            time_major=self.time_major)

        # Decoder
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        sample_id = outputs.sample_id

        # Note: there's a subtle difference here between train and inference.
        # We could have set output_layer when create my_decoder
        #   and shared more code between train and inference.
        # We chose to apply the output_layer to all timesteps for speed:
        #   10% improvements for small models & 20% for larger ones.
        # If memory is a concern, we should apply output_layer per timestep.
        logits = self.output_layer(outputs.rnn_output)

      ## Inference
      else:
        beam_width = hparams.beam_width
        length_penalty_weight = hparams.length_penalty_weight
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer,
              length_penalty_weight=length_penalty_weight)
        else:
          # Helper
          sampling_temperature = hparams.sampling_temperature
          if sampling_temperature > 0.0:
            helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                self.embedding_decoder, start_tokens, end_token,
                softmax_temperature=sampling_temperature,
                seed=hparams.random_seed)
          else:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )

        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state

  # Overrides AttentionModel._build_decoder_cell
  # Allows to disable inference mode.
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length, no_infer=False):
    """Build a RNN cell with attention mechanism that can be used by decoder."""
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture

    if attention_architecture != "standard":
      raise ValueError(
          "Unknown attention architecture %s" % attention_architecture)

    num_units = hparams.num_units
    num_layers = self.num_decoder_layers
    num_residual_layers = self.num_decoder_residual_layers
    beam_width = hparams.beam_width

    dtype = tf.float32

    # Ensure memory is batch-major
    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs

    if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0 and \
        not no_infer:
      memory = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=beam_width)
      source_sequence_length = tf.contrib.seq2seq.tile_batch(
          source_sequence_length, multiplier=beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
          encoder_state, multiplier=beam_width)
      batch_size = self.batch_size * beam_width
    else:
      batch_size = self.batch_size

    attention_mechanism = self.attention_mechanism_fn(
        attention_option, num_units, memory, source_sequence_length, self.mode)

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=self.num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    # Only generate alignment in greedy INFER mode.
    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         beam_width == 0 and not no_infer)
    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=num_units,
        alignment_history=alignment_history,
        output_attention=hparams.output_attention,
        name="attention")

    # TODO(thangluong): do we need num_layers, num_gpus?
    cell = tf.contrib.rnn.DeviceWrapper(cell,
                                        model_helper.get_device_str(
                                            num_layers - 1, self.num_gpus))

    if hparams.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
          cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  # Returns different summaries depending on whether the model is training on
  # a monolingual batch or not.
  def train(self, sess, feed_dict={}):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    if feed_dict[self.mono_batch]:
      train_summary = self.mono_summary
    else:
      train_summary = self.bi_summary
    return sess.run([self.update,
                     self.train_loss,
                     self.predict_count,
                     train_summary,
                     self.global_step,
                     self.word_count,
                     self.batch_size,
                     self.grad_norm,
                     self.learning_rate], feed_dict=feed_dict)

  # Computes the accuracy of how often the max(logits) correctly predicts
  # labels.
  def _compute_accuracy(self, logits, labels, length):
    max_time = self.get_max_time(logits) # time
    mask = tf.sequence_mask(length, max_time, dtype=logits.dtype) # batch x time
    if self.time_major:
      labels = self._transpose_time_major(labels) # time x batch
      mask = self._transpose_time_major(mask) # time x batch

    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32) # time x batch
    equals = tf.cast(tf.equal(predictions, labels), tf.float32)
    time_axis = 0 if self.time_major else 1
    length = tf.cast(length, tf.float32) # batch
    sentence_accuracies = tf.reduce_sum(equals * mask, axis=time_axis) / length
    return tf.reduce_mean(sentence_accuracies)
