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
from collections import defaultdict
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

flags.DEFINE_string("tfidf_candidates_file", None,
                    "Path to TFIDF candidates file.")

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

flags.DEFINE_integer("num_cands", 64, "Number of entity candidates.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")


class Mention(object):
  """A single mention's data."""

  def __init__(self,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               mention_id):
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.mention_id = mention_id

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens[:FLAGS.max_seq_length]]))
    s += "input_ids: %s\n" % (" ".join([str(x) for x in self.input_ids[:FLAGS.max_seq_length]]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids[:FLAGS.max_seq_length]]))
    s += "input_mask: %s\n" % (" ".join([str(x) for x in self.input_mask[:FLAGS.max_seq_length]]))
    s += "mention_id: %s\n" % (" ".join([str(x) for x in self.mention_id[:FLAGS.max_seq_length]]))
    s += "\n"
    return s


class Instance(object):
  """A mention and it's positive single set of features of data."""
  
  def __init__(self,
               mention,
               pos_cand,
               neg_cand):
    self.mention = mention
    self.pos_cand = pos_cand
    self.neg_cand = neg_cand
    

def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    num_cands, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):

    input_ids = []
    input_mask = []
    segment_ids = []
    mention_id = []

    mention_objects = ([instance.mention['object']]
                       + [m['object'] for m in instance.pos_cand]
                       + [m['object'] for m in instance.neg_cand])

    for obj in mention_objects:
      input_ids.extend(obj.input_ids)
      input_mask.extend(obj.input_mask)
      segment_ids.extend(obj.segment_ids)
      mention_id.extend(obj.mention_id)

    assert len(input_ids) == max_seq_length * (2*FLAGS.num_cands + 1)
    assert len(input_mask) == max_seq_length * (2*FLAGS.num_cands + 1)
    assert len(segment_ids) == max_seq_length * (2*FLAGS.num_cands + 1)
    assert len(mention_id) == max_seq_length * (2*FLAGS.num_cands + 1)

    uid_bytes = bytes(instance.mention["mention_id"], "utf-8")
    uid_bytes = list(struct.unpack('=LLLL', uid_bytes)) # will be a list of 4, 32-bit integers
    ## NOTE: to convert back to byte string `struct.pack("=LLLL", *uid_bytes)`

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["mention_id"] = create_int_feature(mention_id)
    features["uid"] = create_int_feature(uid_bytes)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    #if inst_index < 20:
    #  tf.logging.info("*** Example ***")
    #  tf.logging.info("tokens: %s" % " ".join(
    #      [tokenization.printable_text(x) for x in instance.tokens[:FLAGS.max_seq_length]]))

    #  for feature_name in features.keys():
    #    feature = features[feature_name]
    #    values = []
    #    if feature.int64_list.value:
    #      values = feature.int64_list.value
    #    elif feature.float_list.value:
    #      values = feature.float_list.value
    #    tf.logging.info(
    #            "%s: %s" % (feature_name, " ".join([str(x) for x in values[:FLAGS.max_seq_length]])))

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

  tfidf_candidates = {}
  with tf.gfile.GFile(FLAGS.tfidf_candidates_file, "r") as reader:
    while True:
      line = tokenization.convert_to_unicode(reader.readline())
      line = line.strip()
      if not line:
        break
      d = json.loads(line)
      tfidf_candidates[d['mention_id']] = d['tfidf_candidates']

  entity2mention = defaultdict(list)
  for mention in mentions:
    entity2mention[mention['label_document_id']].append(mention)

  vocab_words = list(tokenizer.vocab.keys())

  if FLAGS.split_by_domain:
    instances = {}
  else:
    instances = []

  for i, mention in enumerate(mentions):
    create_mention_object(mention, documents, tfidf_candidates, tokenizer,
                          max_seq_length, vocab_words, rng,
                          is_training=is_training)
    if i > 0 and i % 1000 == 0:
      tf.logging.info("Mention: %d" % i)

  for i, mention in enumerate(mentions):
    instance = create_coref_candidate_sets(
                      mention, mentions, tfidf_candidates, entity2mention)
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


def create_mention_object(
    mention, all_documents, tfidf_candidates, tokenizer, max_seq_length,
    vocab_words, rng, is_training=True):
  """Creates `Mention`s for a single document."""

  # Account for [CLS], [SEP]
  mention_length = max_seq_length - 2

  context_document_id = mention['context_document_id']
  label_document_id = mention['label_document_id']
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

  mention_id = mention['mention_id']
  assert mention_id in tfidf_candidates

  tokens = ['[CLS]'] + mention_context + ['[SEP]']

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  segment_ids = [0]*(len(mention_context) + 2)
  input_mask = [1]*len(input_ids)
  mention_id = [0]*len(input_ids)

  # Update these indices to take [CLS] into account
  new_mention_start = mention_start + 1
  new_mention_end = mention_end + 1

  assert tokens[new_mention_start: new_mention_end+1] == mention_text_tokenized
  for t in range(new_mention_start, new_mention_end+1):
    mention_id[t] = 1

  assert len(input_ids) <= max_seq_length

  tokens = tokens + ['<pad>'] * (max_seq_length - len(tokens))
  input_ids = pad_sequence(input_ids, max_seq_length)
  input_mask = pad_sequence(input_mask, max_seq_length)
  segment_ids = pad_sequence(segment_ids, max_seq_length)
  mention_id = pad_sequence(mention_id, max_seq_length)
  mention['object'] = Mention(tokens=tokens,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              mention_id=mention_id)


def create_coref_candidate_sets(mention, all_mentions, tfidf_candidates,
                                entity2mention):
  num_cands = FLAGS.num_cands
  pos_cand, neg_cand = [], []
  mention_label = mention['label_document_id']

  ### Create the positive candidate set
  # grab other mentions that have the same ground truth entity and have the
  # correct entity in their own candidate set
  pos_cand = [
      m for m in entity2mention[mention_label] 
          if (mention_label in tfidf_candidates[m['mention_id']]
              and m is not mention)
  ]

  # if none of the above exist, grab other mentions that have the same ground
  # truth entity but do not have the correct entity in their own candidate set
  if len(pos_cand) == 0:
    pos_cand = [m for m in entity2mention[mention_label]]

  while len(pos_cand) < num_cands:
    pos_cand.extend(pos_cand)

  pos_cand = pos_cand[:num_cands]

  ### Create the negative candidate set
  neg_cand = random.choices(all_mentions, k=num_cands)
  neg_cand = [m for m in neg_cand
                if (m not in pos_cand and m is not mention)]
  neg_cand = neg_cand[:num_cands//2]
  neg_cand.extend([m for m in random.sample(all_mentions, k=len(all_mentions))
    if (m is not mention 
    and m['label_document_id'] in tfidf_candidates[mention['mention_id']])])

  while len(neg_cand) < num_cands:
    neg_cand.extend(neg_cand)

  neg_cand = neg_cand[:num_cands]

  return Instance(mention, pos_cand, neg_cand)


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
    for corpus in instances:
      output_file = "%s/%s.tfrecord" % (FLAGS.output_file, corpus)
      write_instance_to_example_files(instances[corpus], tokenizer, FLAGS.max_seq_length,
                                      FLAGS.num_cands, [output_file])
  else:
    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.num_cands, [FLAGS.output_file])


if __name__ == "__main__":
  flags.mark_flag_as_required("documents_file")
  flags.mark_flag_as_required("mentions_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
