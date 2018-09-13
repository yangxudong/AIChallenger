"""Create the input data pipeline using `tf.data`"""
import tensorflow as tf

_CSV_COLUMNS = "id,content,location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again,content_ws".split(",")

_CSV_DEFAULTS = [[0], [""], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2],
                 [-2], [-2], [-2], [-2], [-2], [-2], [-2], [""]]


def parse_line(line, vocab):
  columns = tf.decode_csv(line, _CSV_DEFAULTS)
  features = dict(zip(_CSV_COLUMNS, columns))
  content_words = tf.string_split(features.pop("content_ws")).values
  content_ids = vocab.lookup(content_words)
  features.pop("id")
  features.pop("content")
  labels = {k: v + 2 for k, v in features.iteritems()}
  return {"content": content_ids}, labels


def input_fn(path_csv, path_vocab, batch_size, shuffle_buffer_size, num_oov_buckets):
  """Create tf.data Instance from csv file
  Args:
      path_csv: (string) path containing one example per line
      vocab: (tf.lookuptable)
  Returns:
      dataset: (tf.Dataset) yielding list of ids of tokens and labels for each example
  """
  vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=num_oov_buckets)
  # Load txt file, one example per line
  dataset = tf.data.TextLineDataset(path_csv)
  # Convert line into list of tokens, splitting by white space
  dataset = dataset.map(lambda line: parse_line(line, vocab))
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size).repeat()
  dataset = dataset.batch(batch_size).prefetch(1)
  print(dataset.output_types)
  print(dataset.output_shapes)
  return dataset