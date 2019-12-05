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
"""Create training data TF examples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import random
import struct
import tensorflow as tf
from bert import tokenization

from IPython import embed

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("documents_file", None,
                    "Path to documents json file.")

flags.DEFINE_string("mentions_file", None,
                    "Path to mentions json file.")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "is_training", True,
    "Training data")

flags.DEFINE_bool(
    "split_by_domain", False,
    "Split output TFRecords by domain.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("batch_size", 64, "Number of mention-entity pairs in a batch.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

class Mention(object):
  """ A single mention's features. """
  
  def __init__(self,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               uid,
               label_doc_uid):
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.uid = uid
    self.label_doc_uid = label_doc_uid

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "input_ids: %s\n" % (" ".join(
        [str(x) for x in self.input_ids[:FLAGS.max_seq_length]]))
    s += "segment_ids: %s\n" % (" ".join([str(x)
         for x in self.segment_ids[:FLAGS.max_seq_length]]))
    s += "input_mask: %s\n" % (" ".join([str(x)
         for x in self.input_mask[:FLAGS.max_seq_length]]))
    s += "\n"
    return s


class Entity(object):
  """ A single entity's features. """
  def __init__(self,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               uid):
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.uid = uid

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "input_ids: %s\n" % (" ".join(
        [str(x) for x in self.input_ids[:FLAGS.max_seq_length]]))
    s += "segment_ids: %s\n" % (" ".join([str(x)
         for x in self.segment_ids[:FLAGS.max_seq_length]]))
    s += "input_mask: %s\n" % (" ".join([str(x)
         for x in self.input_mask[:FLAGS.max_seq_length]]))
    s += "\n"
    return s


class TrainingInstance(object):
  """A single mention-entity pair."""

  def __init__(self,
               mention_obj,
               entity_obj):
    self.mention_obj = mention_obj
    self.entity_obj = entity_obj


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    batch_size, output_files, is_training):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  # shuffle the instances before we batch them
  random.shuffle(instances)

  if len(instances) % batch_size != 0:
    num_batches = len(instances) // batch_size + 1
    instances.extend(instances[:batch_size])
    instances = instances[:num_batches * batch_size]

  batched_instances = [instances[i*batch_size:(i+1)*batch_size]
                        for i in range(len(instances) // batch_size)]

  total_written = 0

  for batch_index, batch in enumerate(batched_instances):

    batch_mention_input_ids = []
    batch_mention_input_mask = []
    batch_mention_segment_ids = []
    batch_mention_uid_bytes = []
    batch_entity_input_ids = []
    batch_entity_input_mask = []
    batch_entity_segment_ids = []
    batch_entity_uid_bytes = []

    for inst_index, instance in enumerate(batch):
      mention_obj = instance.mention_obj
      entity_obj = instance.entity_obj

      mention_input_ids = mention_obj.input_ids
      mention_input_mask = mention_obj.input_mask
      mention_segment_ids = mention_obj.segment_ids
      # will be a list of 4, 32-bit integers
      mention_uid_bytes = bytes(mention_obj.uid, "utf-8")
      mention_uid_bytes = list(struct.unpack('=LLLL', mention_uid_bytes)) 
      assert mention_obj.uid == struct.pack("=LLLL", *mention_uid_bytes).decode("utf-8")

      entity_input_ids = entity_obj.input_ids
      entity_input_mask = entity_obj.input_mask
      entity_segment_ids = entity_obj.segment_ids
      # will be a list of 4, 32-bit integers
      entity_uid_bytes = bytes(entity_obj.uid, "utf-8")
      entity_uid_bytes = list(struct.unpack('=LLLL', entity_uid_bytes)) 
      assert entity_obj.uid == struct.pack("=LLLL", *entity_uid_bytes).decode("utf-8")

      if is_training:
        assert mention_obj.label_doc_uid == entity_obj.uid

      assert len(mention_input_ids) == max_seq_length
      assert len(mention_input_mask) == max_seq_length
      assert len(mention_segment_ids) == max_seq_length
      assert len(entity_input_ids) == max_seq_length
      assert len(entity_input_mask) == max_seq_length
      assert len(entity_segment_ids) == max_seq_length

      batch_mention_input_ids.extend(mention_input_ids)
      batch_mention_input_mask.extend(mention_input_mask)
      batch_mention_segment_ids.extend(mention_segment_ids)
      batch_mention_uid_bytes.extend(mention_uid_bytes)
      batch_entity_input_ids.extend(entity_input_ids)
      batch_entity_input_mask.extend(entity_input_mask)
      batch_entity_segment_ids.extend(entity_segment_ids)
      batch_entity_uid_bytes.extend(entity_uid_bytes)

      if batch_index < 20 and inst_index < 1:
        tf.logging.info("*** Example ***")
        tf.logging.info("Mention: \n" + str(instance.mention_obj))
        tf.logging.info("Entity: \n" + str(instance.entity_obj))

      total_written += 1

    assert len(batch_mention_input_ids) == batch_size*max_seq_length
    assert len(batch_mention_input_mask) == batch_size*max_seq_length
    assert len(batch_mention_segment_ids) == batch_size*max_seq_length
    assert len(batch_entity_input_ids) == batch_size*max_seq_length
    assert len(batch_entity_input_mask) == batch_size*max_seq_length
    assert len(batch_entity_segment_ids) == batch_size*max_seq_length
    
    features = collections.OrderedDict()
    features["mention_input_ids"] = create_int_feature(batch_mention_input_ids)
    features["mention_input_mask"] = create_int_feature(batch_mention_input_mask)
    features["mention_segment_ids"] = create_int_feature(batch_mention_segment_ids)
    features["mention_uid"] = create_int_feature(batch_mention_uid_bytes)
    features["entity_input_ids"] = create_int_feature(batch_entity_input_ids)
    features["entity_input_mask"] = create_int_feature(batch_entity_input_mask)
    features["entity_segment_ids"] = create_int_feature(batch_entity_segment_ids)
    features["entity_uid"] = create_int_feature(batch_entity_uid_bytes)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    #if inst_index < 20:
    #  tf.logging.info("*** Example ***")
    #  tf.logging.info("Mention: \n" + str(instance.mention_obj))
    #  tf.logging.info("Entity: \n" + str(instance.entity_obj))
    #  #tf.logging.info("tokens: %s" % " ".join(
    #  #    [tokenization.printable_text(x) for x in instance.tokens[:FLAGS.max_seq_length]]))

    #  #for feature_name in features.keys():
    #  #  feature = features[feature_name]
    #  #  values = []
    #  #  if feature.int64_list.value:
    #  #    values = feature.int64_list.value
    #  #  elif feature.float_list.value:
    #  #    values = feature.float_list.value
    #  #  tf.logging.info(
    #  #          "%s: %s" % (feature_name, " ".join([str(x) for x in values[:FLAGS.max_seq_length]])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_training_instances(document_files, mentions_files, tokenizer, max_seq_length,
                              rng, is_training=True):
  """Create `TrainingInstance`s from raw text."""

  documents = {}
  for input_file in document_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        line = line.strip()
        if not line:
          break
        line = json.loads(line)
        documents[line['document_id']] = line

  mentions = []
  for input_file in mentions_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        line = line.strip()
        if not line:
          break
        line = json.loads(line)
        mentions.append(line)

  vocab_words = list(tokenizer.vocab.keys())

  if FLAGS.split_by_domain:
    instances = {}
  else:
    instances = []

  if is_training:
    for i, mention in enumerate(mentions):
      instance = create_instances_from_document(
              mention, documents, tokenizer, max_seq_length,
              vocab_words, rng, is_training=is_training)

      if instance:
        if FLAGS.split_by_domain:
          corpus = mention['corpus']
          if corpus not in instances:
            instances[corpus] = []
          instances[corpus].append(instance)
        else:
          instances.append(instance)

      if i > 0 and i % 1000 == 0:
        tf.logging.info("Instance: %d" % i)
  else:
    mention_objs = []
    entity_objs = []
    for i, mention in enumerate(mentions):
      mention_obj = create_mention_obj(
              mention, documents, tokenizer, max_seq_length,
              vocab_words, rng, is_training=is_training)

      mention_objs.append(mention_obj)

      if i > 0 and i % 1000 == 0:
        tf.logging.info("Mention: %d" % i)

    for i, (_, entity) in enumerate(documents.items()):
      entity_obj = create_entity_obj(
              entity, tokenizer, max_seq_length,
              vocab_words, rng, is_training=is_training)

      entity_objs.append(entity_obj)

      if i > 0 and i % 1000 == 0:
        tf.logging.info("Entity: %d" % i)

    if len(mention_objs) < len(entity_objs):
      while len(mention_objs) < len(entity_objs):
        mention_objs.extend(mention_objs)
      mention_objs = mention_objs[:len(entity_objs)]
    else:
      while len(mention_objs) > len(entity_objs):
        entity_objs.extend(entity_objs)
      entity_objs = entity_objs[:len(mention_objs)]

    for mention_obj, entity_obj in zip(mention_objs, entity_objs):
      instances.append(TrainingInstance(mention_obj, entity_obj))

  if is_training:
    if FLAGS.split_by_domain:
      for corpus in instances:
        rng.shuffle(instances[corpus])
    else:
      rng.shuffle(instances)

  return instances


def get_context_tokens(context_tokens, start_index, end_index, max_tokens, tokenizer):
  start_pos = start_index - max_tokens
  if start_pos < 0:
    start_pos = 0
  prefix = ' '.join(context_tokens[start_pos: start_index])
  suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])
  prefix = tokenizer.tokenize(prefix)
  suffix = tokenizer.tokenize(suffix)
  mention = tokenizer.tokenize(' '.join(context_tokens[start_index: end_index+1]))

  assert len(mention) < max_tokens

  remaining_tokens = max_tokens - len(mention)
  half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))

  mention_context = []

  if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
    prefix_len = half_remaining_tokens
  elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
    prefix_len = remaining_tokens - len(suffix)
  elif len(prefix) < half_remaining_tokens:
    prefix_len = len(prefix)

  if prefix_len > len(prefix):
    prefix_len = len(prefix)

  prefix = prefix[-prefix_len:]

  mention_context = prefix + mention + suffix
  mention_start = len(prefix)
  mention_end = mention_start + len(mention) - 1
  mention_context = mention_context[:max_tokens]
  
  assert mention_start <= max_tokens
  assert mention_end <= max_tokens

  return mention_context, mention_start, mention_end


def pad_sequence(tokens, max_len):
  assert len(tokens) <= max_len
  return tokens + [0]*(max_len - len(tokens))


def create_mention_obj(mention, all_documents, tokenizer, max_seq_length, 
                       vocab_words, rng, is_training=True):
  """Creates `Mention` for single mention"""

  # Account for [CLS], [unused0], [unused1], [SEP]
  mention_length = max_seq_length - 4

  context_document_id = mention['context_document_id']
  start_index = mention['start_index']
  end_index = mention['end_index']

  context_document = all_documents[context_document_id]['text']

  context_tokens = context_document.split()
  extracted_mention = context_tokens[start_index: end_index+1]
  extracted_mention = ' '.join(extracted_mention)
  assert extracted_mention == mention['text']
  mention_text_tokenized = tokenizer.tokenize(mention['text'])

  mention_context, mention_start, mention_end = get_context_tokens(
      context_tokens, start_index, end_index, mention_length, tokenizer)
  
  mention_context.insert(mention_start, '[unused0]')
  mention_context.insert(mention_end+2, '[unused1]')
  mention_context = ['[CLS]'] + mention_context + ['[SEP]']

  input_ids = tokenizer.convert_tokens_to_ids(mention_context)
  segment_ids = [0]*len(input_ids)
  input_mask = [1]*len(input_ids)

  mention_context = mention_context + ['<pad>'] * (max_seq_length - len(mention_context))
  input_ids = pad_sequence(input_ids, max_seq_length)
  segment_ids = pad_sequence(segment_ids, max_seq_length)
  input_mask = pad_sequence(input_mask, max_seq_length)

  mention_obj = Mention(mention_context, input_ids, input_mask,
                        segment_ids, mention['mention_id'],
                        mention['label_document_id'])
  return mention_obj


def create_entity_obj(entity, tokenizer, max_seq_length, vocab_words,
                      rng, is_training=True):
  """Creates `Mention` for single mention"""

  # Account for [CLS], [unused2], [SEP]
  entity_length = max_seq_length - 3

  entity_title_tokens = tokenizer.tokenize(entity['title'])
  entity_text_tokens = tokenizer.tokenize(entity['text'])
  entity_text_tokens = entity_text_tokens[:(entity_length - len(entity_title_tokens))]

  tokens = ['[CLS]'] + entity_title_tokens + ['[unused2]'] \
           + entity_text_tokens + ['[SEP]']

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  segment_ids = [0]*len(input_ids)
  input_mask = [1]*len(input_ids)

  tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))
  input_ids = pad_sequence(input_ids, max_seq_length)
  segment_ids = pad_sequence(segment_ids, max_seq_length)
  input_mask = pad_sequence(input_mask, max_seq_length)

  entity_obj = Entity(tokens, input_ids, input_mask,
                      segment_ids, entity['document_id'])
  return entity_obj


def create_instances_from_document(
    mention, all_documents, tokenizer, max_seq_length,
    vocab_words, rng, is_training=True):
  """Creates `TrainingInstance for single mention."""

  mention_obj = create_mention_obj(mention, all_documents, tokenizer,
                                   max_seq_length, vocab_words, rng,
                                   is_training=True)

  entity_obj = create_entity_obj(all_documents[mention['label_document_id']],
                                 tokenizer, max_seq_length, vocab_words,
                                 rng, is_training=True)

  return TrainingInstance(mention_obj, entity_obj)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  documents_files = []
  for input_pattern in FLAGS.documents_file.split(","):
    documents_files.extend(tf.gfile.Glob(input_pattern))
  mentions_files = []
  for input_pattern in FLAGS.mentions_file.split(","):
    mentions_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in documents_files:
    tf.logging.info("  %s", input_file)
  for input_file in mentions_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      documents_files, mentions_files, tokenizer, FLAGS.max_seq_length,
      rng, is_training=FLAGS.is_training)

  tf.logging.info("*** Writing to output files ***")
  tf.logging.info("  %s", FLAGS.output_file)

  if FLAGS.split_by_domain:
    assert False
    for corpus in instances:
      output_file = "%s/%s.tfrecord" % (FLAGS.output_file, corpus)
      write_instance_to_example_files(instances[corpus], tokenizer, FLAGS.max_seq_length,
                                      FLAGS.batch_size, [output_file], FLAGS.is_training)
  else:
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.batch_size, [FLAGS.output_file],
                                    FLAGS.is_training)


if __name__ == "__main__":
  flags.mark_flag_as_required("documents_file")
  flags.mark_flag_as_required("mentions_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
