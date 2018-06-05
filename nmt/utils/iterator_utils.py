# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""For loading data into NMT models."""
from __future__ import print_function

import collections

import tensorflow as tf

__all__ = ["BatchedInput", "get_iterator", "get_infer_iterator"]


# NOTE(ebrevdo): When we subclass this, instances' __dict__ becomes empty.
class BatchedInput(
    collections.namedtuple("BatchedInput",
                           ("initializer", "source", "source_output", "target_input",
                            "target_output", "source_sequence_length",
                            "target_sequence_length", "mono_initializer",
                            "mono_text_input", "mono_text_output",
                            "mono_text_length", "predicted_source_length",
                            "mono_batch"))):
  pass

def get_mono_iterator(mono_txt_dataset, mono_len_dataset, vocab, sos_id,  eos_id,
      output_buffer_size, batch_size, random_seed=None, num_threads=4):
  mono_dataset = tf.data.Dataset.zip((mono_txt_dataset, mono_len_dataset))
  mono_dataset = mono_dataset.shuffle(output_buffer_size, random_seed)

  mono_dataset = mono_dataset.map(
      lambda txt, pred_src_len: (
          tf.string_split([txt]).values, pred_src_len),
      num_parallel_calls=num_threads)

  # Convert to word ids.
  mono_dataset = mono_dataset.map(
      lambda txt, pred_src_len: (tf.cast(vocab.lookup(txt), tf.int32),
      tf.string_to_number(pred_src_len, out_type=tf.int32)),
      num_parallel_calls=num_threads)

  # Create an input and an output for the decoder
  mono_dataset = mono_dataset.map(
      lambda txt, pred_src_len: (tf.concat(([sos_id], txt), 0), # txt_in
                        tf.concat((txt, [eos_id]), 0),          # txt_out
                        pred_src_len),                          # pred_src_len
      num_parallel_calls=num_threads)

  # Add the text length to the dataset.
  mono_dataset = mono_dataset.map(
      lambda txt_in, txt_out, pred_src_len: (txt_in, txt_out,
      tf.size(txt_in), pred_src_len),
      num_parallel_calls=num_threads)

  batched_mono_dataset = mono_dataset.padded_batch(
      batch_size,
      padded_shapes=(tf.TensorShape([None]),  # monolingual_text_in
                     tf.TensorShape([None]),  # monolingual_text_out
                     tf.TensorShape([]),      # monolingual_text_length
                     tf.TensorShape([])),     # monolingual_src_length_prediction
      padding_values=(eos_id,                 # monolingual_text_in
                      eos_id,                 # mono_lingual_text_out
                      0,                      # monolingual_text_length -- unused
                      0))                     # monolingual_src_length_prediction -- unused
  return batched_mono_dataset.make_initializable_iterator()

def get_infer_iterator(src_dataset,
                       src_vocab_table,
                       batch_size,
                       eos,
                       src_max_len=None):
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)

  if src_max_len:
    src_dataset = src_dataset.map(lambda src: src[:src_max_len])
  # Convert the word strings to ids
  src_dataset = src_dataset.map(
      lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
  # Add in the word counts.
  src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The entry is the source line rows;
        # this has unknown-length vectors.  The last entry is
        # the source row size; this is a scalar.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([])),  # src_len
        # Pad the source sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            0))  # src_len -- unused

  batched_dataset = batching_func(src_dataset)
  batched_iter = batched_dataset.make_initializable_iterator()
  (src_ids, src_seq_len) = batched_iter.get_next()
  return BatchedInput(
      initializer=batched_iter.initializer,
      source=src_ids,
      source_output=tf.zeros([1, 1], dtype=tf.int32),
      target_input=tf.zeros([1, 1], dtype=tf.int32),
      target_output=tf.zeros([1, 1], dtype=tf.int32),
      source_sequence_length=src_seq_len,
      target_sequence_length=tf.ones([1], dtype=tf.int32),
      mono_initializer=None,
      mono_text_input=tf.zeros([1, 1], dtype=tf.int32),
      mono_text_output=tf.zeros([1, 1], dtype=tf.int32),
      mono_text_length=tf.ones([1], dtype=tf.int32),
      predicted_source_length=tf.ones([1], dtype=tf.int32),
      mono_batch=tf.constant(False))

def get_iterator(src_dataset,
                 tgt_dataset,
                 src_vocab_table,
                 tgt_vocab_table,
                 batch_size,
                 sos,
                 eos,
                 random_seed,
                 num_buckets,
                 src_max_len=None,
                 tgt_max_len=None,
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 num_shards=1,
                 shard_index=0,
                 reshuffle_each_iteration=True,
                 mono_datasets=None,
                 mono_batch=tf.constant(False)):
  if not output_buffer_size:
    output_buffer_size = batch_size * 1000
  src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(eos)), tf.int32)
  src_sos_id = tf.cast(src_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_sos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(sos)), tf.int32)
  tgt_eos_id = tf.cast(tgt_vocab_table.lookup(tf.constant(eos)), tf.int32)

  src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

  src_tgt_dataset = src_tgt_dataset.shard(num_shards, shard_index)
  if skip_count is not None:
    src_tgt_dataset = src_tgt_dataset.skip(skip_count)

  src_tgt_dataset = src_tgt_dataset.shuffle(
      output_buffer_size, random_seed, reshuffle_each_iteration)

  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (
          tf.string_split([src]).values, tf.string_split([tgt]).values),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Filter zero length input sequences.
  src_tgt_dataset = src_tgt_dataset.filter(
      lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

  if src_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  if tgt_max_len:
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src, tgt[:tgt_max_len]),
        num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Convert the word strings to ids.  Word strings that are not in the
  # vocab get the lookup table's default_value integer.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                        tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src, tgt: (tf.concat(([src_sos_id], src,), 0),
                        tf.concat((src, [src_eos_id]), 0),
                        tf.concat(([tgt_sos_id], tgt), 0),
                        tf.concat((tgt, [tgt_eos_id]), 0)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)
  # Add in sequence lengths.
  src_tgt_dataset = src_tgt_dataset.map(
      lambda src_in, src_out, tgt_in, tgt_out: (
          src_in, src_out, tgt_in, tgt_out, tf.size(src_in), tf.size(tgt_in)),
      num_parallel_calls=num_parallel_calls).prefetch(output_buffer_size)

  # Bucket by source sequence length (buckets for lengths 0-9, 10-19, ...)
  def batching_func(x):
    return x.padded_batch(
        batch_size,
        # The first three entries are the source and target line rows;
        # these have unknown-length vectors.  The last two entries are
        # the source and target row sizes; these are scalars.
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]),  # src_output
            tf.TensorShape([None]),  # tgt_input
            tf.TensorShape([None]),  # tgt_output
            tf.TensorShape([]),  # src_len
            tf.TensorShape([])),  # tgt_len
        # Pad the source and target sequences with eos tokens.
        # (Though notice we don't generally need to do this since
        # later on we will be masking out calculations past the true sequence.
        padding_values=(
            src_eos_id,  # src
            src_eos_id,  # src_output
            tgt_eos_id,  # tgt_input
            tgt_eos_id,  # tgt_output
            0,  # src_len -- unused
            0))  # tgt_len -- unused

  if num_buckets > 1:

    def key_func(unused_1, unused_2, unused_3, unused_4, src_len, tgt_len):
      # Calculate bucket_width by maximum source sequence length.
      # Pairs with length [0, bucket_width) go to bucket 0, length
      # [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
      # over ((num_bucket-1) * bucket_width) words all go into the last bucket.
      if src_max_len:
        bucket_width = (src_max_len + num_buckets - 1) // num_buckets
      else:
        bucket_width = 10

      # Bucket sentence pairs by the length of their source sentence and target
      # sentence.
      bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
      return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
      return batching_func(windowed_data)

    batched_dataset = src_tgt_dataset.apply(
        tf.contrib.data.group_by_window(
            key_func=key_func, reduce_func=reduce_func, window_size=batch_size))

  else:
    batched_dataset = batching_func(src_tgt_dataset)
  bilingual_iterator = batched_dataset.make_initializable_iterator()

  # Load monolingual data if given.
  if mono_datasets:
    mono_iterator = get_mono_iterator(mono_datasets[0], mono_datasets[1],
        tgt_vocab_table, tgt_sos_id, tgt_eos_id, output_buffer_size, batch_size,
        random_seed=random_seed, num_threads=num_parallel_calls)
    # During bilingual batches give a batch of zeros for monolingual data.
    mono_text_input, mono_text_output, mono_text_length, \
        predicted_source_length = tf.cond(
            mono_batch,
            true_fn=lambda: mono_iterator.get_next(),
            false_fn=lambda: (
                     tf.zeros([batch_size, 1], dtype=tf.int32), # mono_txt_input
                     tf.zeros([batch_size, 1], dtype=tf.int32), # mono_txt_output
                     tf.ones([batch_size], dtype=tf.int32),     # mono_txt_len
                     tf.ones([batch_size], dtype=tf.int32)))    # pred_src_len
    mono_initializer = mono_iterator.initializer
  else:
    mono_text_input, mono_text_output, mono_text_length, predicted_source_length = \
        (tf.zeros([batch_size, 1], dtype=tf.int32), # mono_txt_input
         tf.zeros([batch_size, 1], dtype=tf.int32), # mono_txt_output
         tf.ones([batch_size], dtype=tf.int32),     # mono_txt_len
         tf.ones([batch_size], dtype=tf.int32))    # pred_src_len
    mono_initializer = None

  # If a monolingual batch, just give a batch of zeros for the bilingual data.
  (src_input_ids, src_output_ids, tgt_input_ids, tgt_output_ids, \
      src_seq_len, tgt_seq_len) = tf.cond(mono_batch,
          true_fn=lambda: (
                   tf.zeros([batch_size, 1], dtype=tf.int32), # src
                   tf.zeros([batch_size, 1], dtype=tf.int32), # src_output
                   tf.zeros([batch_size, 1], dtype=tf.int32), # tgt_input
                   tf.zeros([batch_size, 1], dtype=tf.int32), # tgt_output
                   tf.ones([batch_size], dtype=tf.int32),     # src_len
                   tf.ones([batch_size], dtype=tf.int32)),    # tgt_len
          false_fn=lambda: bilingual_iterator.get_next())

  return BatchedInput(
      initializer=bilingual_iterator.initializer,
      source=src_input_ids,
      source_output=src_output_ids,
      target_input=tgt_input_ids,
      target_output=tgt_output_ids,
      source_sequence_length=src_seq_len,
      target_sequence_length=tgt_seq_len,
      mono_initializer=mono_initializer,
      mono_text_input=mono_text_input,
      mono_text_output=mono_text_output,
      mono_text_length=mono_text_length,
      predicted_source_length=predicted_source_length,
      mono_batch=mono_batch)
