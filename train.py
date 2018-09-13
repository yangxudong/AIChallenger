#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Train the model"""
import os
import tensorflow as tf
from model.Params import Params
from model.TextCNN import TextCNN
from model.input_fn import input_fn

flags = tf.app.flags
flags.DEFINE_string("data_dir", "data", "Directory containing the dataset.")
flags.DEFINE_string("model_dir", "experiments/TextCNN", "Base directory for the model.")
flags.DEFINE_integer("save_checkpoints_steps", 2000, "Save checkpoints every this many steps")
flags.DEFINE_integer("train_steps", 10000, "Number of (global) training steps to perform")
FLAGS = flags.FLAGS

def main(unused_argv):
  json_path = os.path.join(FLAGS.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  params = Params(json_path)
  # Load the parameters from the dataset, that gives the size etc. into params
  json_path = os.path.join(FLAGS.data_dir, 'dataset_params.json')
  assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
  params.update(json_path)

  path_words = os.path.join(FLAGS.data_dir, 'words.txt')
  path_train = os.path.join(FLAGS.data_dir, 'train.csv')
  path_eval = os.path.join(FLAGS.data_dir, 'valid.csv')
  path_test = os.path.join(FLAGS.data_dir, 'testa.csv')
  print("train set:", path_train)
  print("valid set:", path_eval)
  print("test set:", path_test)

  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
  if params.model == "TextCNN":
    estimator = TextCNN(params, model_dir=FLAGS.model_dir, config=config)
    train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: input_fn(path_train, path_words, params, params.shuffle_buffer_size),
      max_steps=FLAGS.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: input_fn(path_eval, path_words, params, 0),
      throttle_secs=300
    )
    print("before train and evaluate")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("after train and evaluate")
    #test_input_fn = lambda: input_fn(path_test, path_words, params.batch_size, 0, params.num_oov_buckets)
    #predictions = estimator.predict(test_input_fn)


if __name__ == '__main__':
  os.environ["CUDA_VISIBLE_DEVICES"] = ""
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)

