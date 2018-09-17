#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Train the model"""
import os
import pandas as pd
import tensorflow as tf
from model.Params import Params
from model.TextCNN import TextCNN
from model.input_fn import input_fn

flags = tf.app.flags
flags.DEFINE_string("data_dir", "data", "Directory containing the dataset.")
flags.DEFINE_string("model_dir", "experiments/TextCNN", "Base directory for the model.")
flags.DEFINE_string("gpu_id", "0", "which gpu to use.")
flags.DEFINE_integer("save_checkpoints_steps", 3000, "Save checkpoints every this many steps")
flags.DEFINE_bool("train", True, "Whether to train and evaluation")
flags.DEFINE_bool("predict", True, "Whether to predict")
FLAGS = flags.FLAGS

OUTPUT_CSV_COLUMNS = "content,location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again".split(",")

def main(unused_argv):
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id
  json_path = os.path.join(FLAGS.model_dir, 'params.json')
  assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
  print(json_path)
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
  if params.model.startswith("TextCNN"):
    estimator = TextCNN(params, model_dir=FLAGS.model_dir, config=config, optimizer=params.optimizer if "optimizer" in params else None)
  if FLAGS.train:
    train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: input_fn(path_train, path_words, params, params.shuffle_buffer_size),
      max_steps=params.train_steps
    )
    eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: input_fn(path_eval, path_words, params, 0),
      throttle_secs=600
    )
    print("before train and evaluate")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("after train and evaluate")
    #inputs = {"content": tf.placeholder(shape=[None, params.sentence_max_len], dtype=tf.int32),
    #  "id": tf.placeholder(shape=[None, 1], dtype=tf.int32)}
    #estimator.export_savedmodel(
    #  export_dir_base=FLAGS.model_dir, serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(inputs))
  if FLAGS.predict:
    params.batch_size = 1
    test_input_fn = lambda: input_fn(path_test, path_words, params, 0)
    predictions = estimator.predict(test_input_fn)
    result = pd.DataFrame(predictions)
    output_path = os.path.join(FLAGS.model_dir, params.model + '_result.csv')
    result.to_csv(output_path, index_label="id", columns=OUTPUT_CSV_COLUMNS)


if __name__ == '__main__':
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)

