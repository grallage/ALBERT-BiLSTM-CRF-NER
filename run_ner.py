# coding=utf-8

"""
@Author:Lynn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tensorflow as tf
import collections
import tokenization
import pickle
import tensorflow as tf
from lstm_crf_layer import BLSTM_CRF
import codecs
from tensorflow.contrib.layers.python.layers import initializers

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .txt files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "ner", "The name of the task to train. Only supports NER task.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

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
flags.DEFINE_bool("use_one_hot_embeddings", False, "Whether to use one hot embeddings.")

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

flags.DEFINE_float("dropout_rate", 0.5, "dropout_rate.")

flags.DEFINE_integer("lstm_size", 128, "lstm_size.")

flags.DEFINE_string("cell", 'lstm', "cell.")

flags.DEFINE_integer("num_layers", 1, "num_layers.")

flags.DEFINE_integer("batch_size", 64, "batch_size.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text: string. The untokenized text of the sequence.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    # not need
    # self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_data(cls, input_file):
    """Reads a BIO data."""
    with open(input_file) as f:
      lines = []
      words = []
      labels = []
      for line in f:
        contends = line.strip()
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        if contends.startswith("-DOCSTART-"):
          words.append('')
          continue
        if len(contends) == 0:
          l = ' '.join([label for label in labels if len(label) > 0])
          w = ' '.join([word for word in words if len(word) > 0])
          lines.append([l, w])
          words = []
          labels = []
          continue
        words.append(word)
        labels.append(label)
      return lines

class NerProcessor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "train.txt")), "train"
    )

  def get_dev_examples(self, data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
    )

  def get_test_examples(self,data_dir):
    return self._create_example(
      self._read_data(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
      labels = ["O", 'B-TIM', 'I-TIM', "B-PER", "I-PER", "B-ORG"
      , "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
      with codecs.open(os.path.join(FLAGS.output_dir, 'label_list.pkl'), 'wb') as rf:
        pickle.dump(labels, rf)
      return labels

  def _create_example(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(InputExample(guid=guid, text=text, label=label))
    return examples


def write_tokens(tokens, mode):
  if mode == "test":
    path = os.path.join(FLAGS.output_dir, "token_" + mode + ".txt")
    wf = codecs.open(path, 'a', encoding='utf-8')
    for token in tokens:
      if token != "**NULL**":
        wf.write(token + '\n')
    wf.close()

def convert_single_example(ex_index, example, label_map, max_seq_length,
 tokenizer, mode):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  textlist = example.text.split(' ')
  labellist = example.label.split(' ')
  tokens_a = []
  labels = []
  
  for i, word in enumerate(textlist):
    token = tokenizer.tokenize(word)  
    tokens_a.extend(token)
    label_1 = labellist[i]
  
    for m in range(len(token)):
      if m == 0:
        labels.append(label_1)
      else:
        labels.append("X")

  # Account for [CLS] and [SEP] with "- 2"
  if len(tokens_a) >= max_seq_length - 2:
    tokens_a = tokens_a[0:(max_seq_length - 2)]
    labels = labels[0:(max_seq_length - 2)]

  tokens = []
  segment_ids = []
  label_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  label_ids.append(label_map["[CLS]"])

  for i, token in enumerate(tokens_a):
    tokens.append(token)
    segment_ids.append(0)
    label_ids.append(label_map[labels[i]])

  tokens.append("[SEP]")
  segment_ids.append(0)
  label_ids.append(label_map["[SEP]"])

  input_ids = tokenizer.convert_tokens_to_ids(tokens)
  input_mask = [1] * len(input_ids)
  #label_mask = [1] * len(input_ids)
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)
    label_ids.append(0)
    tokens.append("**NULL**")
    #label_mask.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length
  assert len(label_ids) == max_seq_length
  #assert len(label_mask) == max_seq_length

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
      [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))    

  feature = InputFeatures(
    input_ids=input_ids,
    input_mask=input_mask,
    segment_ids=segment_ids,
    label_ids=label_ids,
    #label_mask = label_mask
  )
  write_tokens(tokens, mode)
  return feature

def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file, mode=None):
  """Convert a set of `InputExample`s to a TFRecord file."""

  label_map = {}
  for (i, label) in enumerate(label_list, 1):
    label_map[label] = i

  if not os.path.exists(os.path.join(FLAGS.output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_map,
                                     max_seq_length, tokenizer, mode)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature(feature.label_ids)
    # features["is_real_example"] = create_int_feature(
    #     [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
      # "is_real_example": tf.FixedLenFeature([], tf.int64),
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
    tf.logging.info('*** input_fn, input_file %s 测试input_fn %s' % (input_file,d))
    return d

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0, lstm_size=1, cell='lstm', num_layers=1):

    model = modeling.AlbertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # embedding.shape = [batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    # 算序列真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度
    # add CRF output layer
    blstm_crf = BLSTM_CRF(embedded_chars=embedding, hidden_unit=lstm_size, cell_type=cell, num_layers=num_layers,
                          dropout_rate=dropout_rate, initializers=initializers, num_labels=num_labels,
                          seq_length=max_seq_length, labels=labels, lengths=lengths, is_training=is_training)
    rst = blstm_crf.add_blstm_crf_layer(crf_only=True)
    return rst


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, #use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Here is model_fn, Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s, type = %s" % (name, features[name].shape, type(features[name])))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False, FLAGS.dropout_rate, FLAGS.lstm_size, FLAGS.cell, FLAGS.num_layers)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    # scaffold_fn = None

    # load albert model checkpoint
    if init_checkpoint:
      (assignment_map, initialized_variable_names) = \
        modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.logging.info('*** Here is model_fn, assignment_map = %s ***' % assignment_map)
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
      use_tpu = False
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          )
    elif mode == tf.estimator.ModeKeys.EVAL:
      # 针对NER ,进行了修改
      def metric_fn(label_ids, pred_ids):
        return {
        "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
        }

      eval_metrics = metric_fn(label_ids, pred_ids)
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metric_ops=eval_metrics
      )
    else:
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_ids
      )
    tf.logging.info('type output_spec = %s, mode= %s' % (type(output_spec), mode))
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "ner": NerProcessor
  }

  # Step 1. check pre-train model and some config parameters
  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)  

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.AlbertConfig.from_json_file(FLAGS.bert_config_file)  

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  # Step 2. define raw data processor and some tf config
  processor = processors[task_name]()  

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # A ProtocolMessage
  session_config = tf.ConfigProto(
    log_device_placement=False,
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0,
    allow_soft_placement=True)

  run_config = tf.estimator.RunConfig(
    model_dir=FLAGS.output_dir,
    save_summary_steps=500,
    save_checkpoints_steps=500,
    session_config=session_config
  )

  # InputExample list, each item contain InputExample instance
  train_examples = None
  eval_examples = None

  num_train_steps = None
  num_warmup_steps = None

  # Step 3. convert raw data to InputExample object
  if FLAGS.do_train and FLAGS.do_eval:  
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(len(train_examples) * 1.0 / FLAGS.batch_size * FLAGS.num_train_epochs)

    if num_train_steps < 1:
      raise AttributeError('training data is so small...')

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", FLAGS.batch_size)

  # Step 4. define tf model_fn and tf estimator
  label_list = processor.get_labels()

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list) + 1,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  params = {
    'batch_size': FLAGS.batch_size
  }

  estimator = tf.estimator.Estimator(
    model_fn,
    params=params,
    config=run_config)

  # Step 5. training data
  if FLAGS.do_train and FLAGS.do_eval:    
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    # InputExample list convert to tf_record file
    if not os.path.exists(train_file):
      file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    del train_examples

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
      file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    del eval_examples
    
    # read train data from tf_record file
    train_input_fn = file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

    eval_input_fn = file_based_input_fn_builder(
      input_file=eval_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

    early_stopping_hook = tf.estimator.experimental.stop_if_no_decrease_hook(
      estimator=estimator,
      metric_name='loss',
      max_steps_without_decrease=num_train_steps,
      eval_dir=None,
      min_steps=0,
      run_every_secs=None,
      run_every_steps=FLAGS.save_checkpoints_steps)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
      hooks=[early_stopping_hook])

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  # Step 6. predict data
  if FLAGS.do_predict:
    token_path = os.path.join(FLAGS.output_dir, "token_test.txt")
    if os.path.exists(token_path):
      os.remove(token_path)

    with codecs.open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as rf:
      label2id = pickle.load(rf)
      id2label = {value:key for key,value in label2id.items()}
    
    predict_examples = processor.get_test_examples(FLAGS.data_dir)

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file, mode="test")
                            
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    output_predict_file = os.path.join(FLAGS.output_dir, "label_test.txt")

    with open(output_predict_file,'w') as writer:
      for prediction in result:
        output_line = "\n".join(id2label[id] for id in prediction if id!=0) + "\n"
        writer.write(output_line)

  tf.logging.info("***** finish *****")


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()