#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Train the model"""
import os
import tensorflow as tf
import json

flags = tf.app.flags
flags.DEFINE_string("data_dir", "data", "Directory containing the dataset.")
flags.DEFINE_string("model_dir", "experiments/base_model", "Base directory for the model.")

FLAGS = flags.FLAGS

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

def main(unused_argv):
  path_words = os.path.join(FLAGS.data_dir, 'words.txt')
  num_oov_buckets = 1
  # Load Vocabularies
  words = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=num_oov_buckets)


if __name__ == '__main__':
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)

