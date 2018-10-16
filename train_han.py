#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Train the model"""
import os
import pandas as pd
import tensorflow as tf
from model.Params import Params
from model.HAN import HAN
from model.doc_input_fn import input_fn

flags = tf.app.flags
flags.DEFINE_string("data_dir", "data", "Directory containing the dataset.")
flags.DEFINE_string("model_dir", "experiments/HAN", "Base directory for the model.")
flags.DEFINE_string("gpu", "0", "which gpu to use.")
flags.DEFINE_string("target_prefix", "price_level", "the prefix of target name.")
flags.DEFINE_integer("save_checkpoints_steps", 3000, "Save checkpoints every this many steps")
flags.DEFINE_integer("keep_checkpoint_max", 20, "how many checkpoints will be keep")
flags.DEFINE_integer("throttle_secs", 300, "evaluation time span in seconds")
flags.DEFINE_bool("train", True, "Whether to train and evaluation")
flags.DEFINE_bool("predict", True, "Whether to predict")
FLAGS = flags.FLAGS

TARGETS = "location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again".split(",")

def train_and_predict(label):
  print("start train_and_predict target: " + label)
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
  model_dir = os.path.join(FLAGS.model_dir, label)
  config = tf.estimator.RunConfig(model_dir=model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps, keep_checkpoint_max=FLAGS.keep_checkpoint_max)
  estimator = HAN(params, model_dir=model_dir, config=config, optimizer=params.optimizer if "optimizer" in params else None)
  if FLAGS.train:
    train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: input_fn(path_train, path_words, label, params, params.shuffle_buffer_size),
      max_steps=params.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: input_fn(path_eval, path_words, label, params, 0),
      throttle_secs=FLAGS.throttle_secs
    )
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  if FLAGS.predict:
    params.batch_size = 1
    test_input_fn = lambda: input_fn(path_test, path_words, label, params, 0)
    predictions = estimator.predict(test_input_fn)
    result = pd.DataFrame(predictions)
    output_path = os.path.join(model_dir, label + '_result.csv')
    result.to_csv(output_path, columns=["class_ids"])
  print("finished train and predict of target: " + label)


def main(unused_argv):
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  targets = [t for t in TARGETS if t.startswith(FLAGS.target_prefix)]
  print("targets:", targets)
  for target in targets:
    train_and_predict(target)

if __name__ == '__main__':
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
