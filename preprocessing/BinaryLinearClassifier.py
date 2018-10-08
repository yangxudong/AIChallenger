import tensorflow as tf
import os
from model.Params import Params

_CSV_COLUMNS = "content,id,location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again,content_ws".split(",")

_CSV_DEFAULTS = [[""], [0], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2],
                 [-2], [-2], [-2], [-2], [-2], [-2], [-2], [""]]

flags = tf.app.flags
flags.DEFINE_string("data_dir", "..\\data", "Directory containing the dataset.")
flags.DEFINE_string("model_dir", "..\\experiments\\TextCNN", "Base directory for the model.")
flags.DEFINE_string("gpu", "0", "which gpu to use.")
flags.DEFINE_integer("save_checkpoints_steps", 1000, "Save checkpoints every this many steps")
flags.DEFINE_integer("throttle_secs", 300, "evaluation time span in seconds")
flags.DEFINE_bool("train", True, "Whether to train and evaluation")
flags.DEFINE_bool("predict", True, "Whether to predict")
FLAGS = flags.FLAGS

def parse_line(line, vocab, max_len, target):
  columns = tf.decode_csv(line, _CSV_DEFAULTS, field_delim=',')
  features = dict(zip(_CSV_COLUMNS, columns))
  content_words = tf.string_split([features.pop("content_ws")]).values
  content_length = tf.size(content_words)
  content_words = tf.slice(content_words, [0], [tf.minimum(content_length, max_len)])
  content_ids = vocab.lookup(content_words)
  return content_ids, features[target]


def transform_label(x, y):
  yy = tf.where(tf.equal(y, 1), y, 0)
  return x, yy


def input_fn(path_csv, path_vocab, target, params, shuffle_buffer_size):
  """Create tf.data Instance from csv file
  Args:
      path_csv: (string) path containing one example per line
      vocab: (tf.lookuptable)
  Returns:
      dataset: (tf.Dataset) yielding list of ids of tokens and labels for each example
  """
  #vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=params.num_oov_buckets)
  #params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))
  # Load txt file, one example per line
  dataset = tf.data.TextLineDataset(path_csv)
  # Convert line into list of tokens, splitting by white space
  dataset = dataset.skip(1).map(lambda line: parse_line(line, path_vocab, params.sentence_max_len, target)) # skip the header
  dataset = dataset.filter(lambda x, y: tf.logical_or(tf.equal(y, -1), tf.equal(y, 1)))
  dataset = dataset.map(lambda x, y: transform_label(x, y))
  #if shuffle_buffer_size > 0:
  #  dataset = dataset.shuffle(shuffle_buffer_size).repeat()
  # Create batches and pad the sentences of different length
  padded_shapes = ([params.sentence_max_len], [])
  #padding_values = (params.id_pad_word, 0)
  padding_values = (tf.cast(0, tf.int64), 0)
  dataset = dataset.padded_batch(params.batch_size, padded_shapes, padding_values).prefetch(1)
  print(dataset.output_types)
  print(dataset.output_shapes)
  return dataset


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

  vocab = tf.contrib.lookup.index_table_from_file(path_words, num_oov_buckets=params.num_oov_buckets)
  dataset = input_fn(path_eval, vocab, "service_wait_time", params, params.shuffle_buffer_size)
  iterator = dataset.make_initializable_iterator()
  iter_init_op = iterator.initializer
  one_element = iterator.get_next()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    #params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))
    # print(sess.run(mask))
    sess.run(iter_init_op)
    for i in range(4):
      print(sess.run(one_element))

if __name__ == '__main__':
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
  #tf.logging.set_verbosity(tf.logging.INFO)
  #tf.app.run(main=main)
  main(1)
