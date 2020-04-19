# coding=utf-8

"""
@Author:Lynn
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
# import optimization
import tensorflow as tf
# import collections
import tokenization
import pickle
import tensorflow as tf
# from lstm_crf_layer import BLSTM_CRF
import codecs
# from tensorflow.contrib.layers.python.layers import initializers
import ner_utils

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


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  ner_utils.FLAGS = FLAGS

  processors = {
      "ner": ner_utils.NerProcessor
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

  model_fn = ner_utils.model_fn_builder(
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
      ner_utils.file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    del train_examples

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
      ner_utils.file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
    del eval_examples
    
    # read train data from tf_record file
    train_input_fn = ner_utils.file_based_input_fn_builder(
      input_file=train_file,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)

    eval_input_fn = ner_utils.file_based_input_fn_builder(
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
    ner_utils.file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file, mode="test")
                            
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = ner_utils.file_based_input_fn_builder(
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