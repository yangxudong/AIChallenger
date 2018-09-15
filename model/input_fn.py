"""Create the input data pipeline using `tf.data`"""
import tensorflow as tf

_CSV_COLUMNS = "content,id,location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again,content_ws".split(",")

_CSV_DEFAULTS = [[""], [0], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2],
                 [-2], [-2], [-2], [-2], [-2], [-2], [-2], [""]]


def parse_line(line, vocab, max_len):
  columns = tf.decode_csv(line, _CSV_DEFAULTS, field_delim=',')
  features = dict(zip(_CSV_COLUMNS, columns))
  content_words = tf.string_split([features.pop("content_ws")]).values
  content_length = tf.size(content_words)
  content_words = tf.slice(content_words, [0], [tf.minimum(content_length, max_len)])
  content_ids = vocab.lookup(content_words)
  #sample_id = features.pop("id")
  #features.pop("content")
  features["content"] = content_ids
  ignore_columns = set(["content", "id"])
  for k in features.keys():
    if k in ignore_columns: continue
    features[k] = features[k] + 2
  #labels = {k: v + 2 for k, v in features.iteritems() if k not in ignore_columns}
  return features, features["id"]


def input_fn(path_csv, path_vocab, params, shuffle_buffer_size):
  """Create tf.data Instance from csv file
  Args:
      path_csv: (string) path containing one example per line
      vocab: (tf.lookuptable)
  Returns:
      dataset: (tf.Dataset) yielding list of ids of tokens and labels for each example
  """
  vocab = tf.contrib.lookup.index_table_from_file(path_vocab, num_oov_buckets=params.num_oov_buckets)
  params.id_pad_word = vocab.lookup(tf.constant(params.pad_word))
  # Load txt file, one example per line
  dataset = tf.data.TextLineDataset(path_csv)
  # Convert line into list of tokens, splitting by white space
  dataset = dataset.skip(1).map(lambda line: parse_line(line, vocab, params.sentence_max_len)) # skip the header
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size).repeat()
  # Create batches and pad the sentences of different length
  ignore_columns = set(["content", "content_ws"])
  feature_shapes = {k:tf.TensorShape([]) for k in _CSV_COLUMNS if not k in ignore_columns}
  feature_shapes["content"] = tf.TensorShape([params.sentence_max_len])
  pad_values = {k:-2 for k in _CSV_COLUMNS if not k in ignore_columns}
  pad_values["content"] = params.id_pad_word
  padded_shapes = (feature_shapes, [])
  padding_values = (pad_values, 0)
  dataset = dataset.padded_batch(params.batch_size, padded_shapes, padding_values).prefetch(1)
  #print(dataset.output_types)
  #print(dataset.output_shapes)
  return dataset
