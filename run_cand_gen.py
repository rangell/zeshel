# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Entity Linking training/eval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import pickle
import struct
import modeling
import numpy as np
import bert
from bert import optimization
from bert import tokenization
from bert.modeling import create_initializer
import tensorflow as tf

from IPython import embed


NUM_UID_BYTES = 4


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "output_dim", 128,
    "The dimension of the output representations for each mention.")

flags.DEFINE_integer(
    "num_cands", 64,
    "Size of positive and negative candidate sets.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_string("eval_domain", 'val_coronation', "Eval domain.")

flags.DEFINE_string("output_eval_file", '/tmp/eval.txt', "Eval output file.")

flags.DEFINE_string("output_rep_file", '/tmp/reps.pkl', "Eval output file.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_integer(
    "num_train_examples", 800,
    "Number of training examples.")


def file_based_input_fn_builder(input_file, num_cands, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "mention_input_ids": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
      "mention_input_mask": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
      "mention_segment_ids": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
      "mention_uid": tf.FixedLenFeature([num_cands, NUM_UID_BYTES], tf.int64),
      "entity_input_ids": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
      "entity_input_mask": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
      "entity_segment_ids": tf.FixedLenFeature([num_cands, seq_length], tf.int64),
      "entity_uid": tf.FixedLenFeature([num_cands, NUM_UID_BYTES], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_zeshel_model(bert_config, output_dim, num_cands, is_training,
        mention_input_ids, mention_input_mask, mention_segment_ids,
        entity_input_ids, entity_input_mask, entity_segment_ids,
        use_one_hot_embeddings):
  """Creates a classification model."""

  batch_size = mention_input_ids.shape[0].value
  seq_len = mention_input_ids.shape[-1].value

  input_ids = tf.concat([mention_input_ids, entity_input_ids], 1)
  segment_ids = tf.concat([mention_segment_ids, entity_segment_ids], 1)
  input_mask = tf.concat([mention_input_mask, entity_input_mask], 1)

  input_ids = tf.reshape(input_ids, [-1, seq_len])
  input_mask = tf.reshape(input_mask, [-1, seq_len])
  segment_ids = tf.reshape(segment_ids, [-1, seq_len])

  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  output_layer = model.get_pooled_output()
  output_layer = tf.reshape(output_layer, [batch_size, 2*num_cands, -1])

  hidden_size = output_layer.shape[-1].value

  output_dim = 128
  
  output_weights = tf.get_variable(
      "output_weights", [output_dim, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [output_dim], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    reps = tf.matmul(output_layer, output_weights, transpose_b=True)
    reps = tf.nn.bias_add(reps, output_bias)

    reps = tf.nn.l2_normalize(reps, axis=-1)

    mention_reps = tf.reshape(reps[:, :num_cands, :], [batch_size, num_cands, output_dim])
    entity_reps = tf.reshape(reps[:, num_cands:, :], [batch_size, num_cands, output_dim])

    scores = tf.matmul(mention_reps, entity_reps, transpose_b=True)
    pos_scores = tf.linalg.diag_part(scores)

    no_diag_mask = np.ones(batch_size * num_cands**2, dtype=np.bool).reshape(batch_size, num_cands, num_cands)
    no_diag_mask[:, np.arange(num_cands), np.arange(num_cands)] = False

    scores_no_diag = tf.boolean_mask(scores, no_diag_mask)

    per_example_loss = (tf.reduce_logsumexp(scores_no_diag, axis=-1)) + 1.0 - pos_scores

    loss = tf.reduce_mean(per_example_loss)

    return (loss,
            per_example_loss,
            mention_reps,
            entity_reps,
            scores)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, output_dim,
                     num_cands, use_tpu, use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    mention_input_ids = features["mention_input_ids"]
    mention_input_mask = features["mention_input_mask"]
    mention_segment_ids = features["mention_segment_ids"]
    mention_uids = features["mention_uid"]
    entity_input_ids = features["entity_input_ids"]
    entity_input_mask = features["entity_input_mask"]
    entity_segment_ids = features["entity_segment_ids"]
    entity_uids = features["entity_uid"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, mention_rep, entity_rep, scores) = create_zeshel_model(
        bert_config, output_dim, num_cands, is_training,
        mention_input_ids, mention_input_mask, mention_segment_ids,
        entity_input_ids, entity_input_mask, entity_segment_ids,
        use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = bert.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"mention_uid": mention_uids, "mention_rep": mention_rep,
                       "entity_uid": entity_uids, "entity_rep": entity_rep,
                       "scores": scores},
          loss=total_loss,
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = bert.modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_examples = FLAGS.num_train_examples
    num_train_steps = int(
        num_train_examples / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      output_dim=FLAGS.output_dim,
      num_cands=FLAGS.num_cands,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.data_dir, "train.tfrecord")
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Train file= %s", train_file)
    tf.logging.info("  Num examples = %d", num_train_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:

    eval_file = os.path.join(FLAGS.data_dir, FLAGS.eval_domain + ".tfrecord")

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      eval_steps = 0
      for fn in [eval_file]:
        for record in tf.python_io.tf_record_iterator(fn):
          eval_steps += 1

      eval_steps = int(eval_steps // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    tf.logging.info("################## STARTING EVAL #######################")

    result = estimator.evaluate(
        input_fn=eval_input_fn,
        steps=eval_steps,
        name=FLAGS.eval_domain)

    output_eval_file = FLAGS.output_eval_file
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_file = os.path.join(FLAGS.data_dir, FLAGS.eval_domain + ".tfrecord")

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        num_cands=FLAGS.num_cands,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    result = estimator.predict(input_fn=predict_input_fn)

    mention_reps = {}
    entity_reps = {}
    for prediction in result:
      for i in range(FLAGS.num_cands):
        mention_uid = struct.pack("=LLLL", *list(prediction["mention_uid"][i])).decode("utf-8")
        mention_reps[mention_uid] = prediction["mention_rep"][i]
        entity_uid = struct.pack("=LLLL", *list(prediction["entity_uid"][i])).decode("utf-8")
        entity_reps[entity_uid] = prediction["entity_rep"][i]

    reps = {'entity_reps': entity_reps, 'mention_reps': mention_reps}

    output_rep_file = FLAGS.output_rep_file
    with tf.gfile.GFile(output_rep_file, "wb") as f:
        pickle.dump(reps, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
