import tensorflow as tf
import nmt.utils.misc_utils as utils

from nmt import model_helper

# Creates an RNN language model and returns its raw outputs. If time_major is
# set it expects embedding inputs in time_major. If z_sample is given it will
# initialize the LM with tanh(W z_sample).
def language_model(embeddings, sequence_length, hparams, mode, single_cell_fn,
    time_major, batch_size, z_sample=None):

  with tf.variable_scope("language_model") as scope:
    # Use decoder cell options.
    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        num_layers=hparams.num_lm_layers,
        num_residual_layers=hparams.num_decoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=hparams.num_gpus,
        mode=mode,
        single_cell_fn=single_cell_fn)

    # Use a zero initial state or tanh(Wz) if provided (VAEJointModel).
    if z_sample is not None:
      utils.print_out("  initializing generative LM with tanh(Wz)")
      init_state_val = tf.tanh(tf.layers.dense(z_sample, hparams.num_units))
      init_state = make_initial_state(init_state_val, hparams.unit_type)
    else:
      utils.print_out("  initializing generative LM with zeros.")
      init_state = cell.zero_state(batch_size, scope.dtype)

    # Run the RNN language model.
    helper = tf.contrib.seq2seq.TrainingHelper(
        embeddings,
        sequence_length,
        time_major=time_major)
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        initial_state=init_state)
    lm_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
        decoder,
        output_time_major=time_major,
        impute_finished=True,
        scope=scope)

    return lm_outputs

def make_initial_state(initial_state_val, unit_type):
  if unit_type == "lstm":
    initial_state = tf.contrib.rnn.LSTMStateTuple(initial_state_val,
        tf.zeros_like(initial_state_val))
    return initial_state
  elif unit_type == "gru":
    return initial_state_val
  else:
    raise ValueError("Unknown unit_type: %s for make_initial_state" % unit_type)
