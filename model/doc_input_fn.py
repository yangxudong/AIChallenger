"""Create the input data pipeline using `tf.data`"""
import tensorflow as tf

_CSV_COLUMNS = "content,id,location_traffic_convenience,location_distance_from_business_district,location_easy_to_find,\
service_wait_time,service_waiters_attitude,service_parking_convenience,service_serving_speed,\
price_level,price_cost_effective,price_discount,\
environment_decoration,environment_noise,environment_space,environment_cleaness,\
dish_portion,dish_taste,dish_look,dish_recommendation,\
others_overall_experience,others_willing_to_consume_again,sentences,sentence_len".split(",")

_CSV_DEFAULTS = [[""], [0], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2], [-2],
                 [-2], [-2], [-2], [-2], [-2], [-2], [-2], [""], [""]]

def parse_line(line, vocab, pad_word, label):
  columns = tf.decode_csv(line, _CSV_DEFAULTS, field_delim=',')
  features = dict(zip(_CSV_COLUMNS, columns))
  
  sentences = tf.string_split([features["sentences"]], ":").values
  words = tf.string_split(sentences)
  word_mat = tf.sparse_tensor_to_dense(words, default_value=pad_word)
  word_ids = vocab.lookup(word_mat)
  sentence_len = tf.string_to_number(tf.string_split([features["sentence_len"]], ":").values, out_type=tf.int32)
  sentence_num = tf.size(sentence_len)
  out_features = {"content": word_ids, "sentence_len": sentence_len, "sentence_num": sentence_num}
  return out_features, features[label] + 2


def input_fn(path_csv, path_vocab, label, params, shuffle_buffer_size):
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
  dataset = dataset.skip(1).map(lambda line: parse_line(line, vocab, params.pad_word, label)) # skip the header
  if shuffle_buffer_size > 0:
    dataset = dataset.shuffle(shuffle_buffer_size).repeat()
  # Create batches and pad the sentences of different length
  feature_shapes = {"content": tf.TensorShape([params.max_sentence_num, params.max_sentence_len]),
                    "sentence_len": tf.TensorShape([params.max_sentence_num]),
                    "sentence_num": tf.TensorShape([])}
  pad_values = {"content": params.id_pad_word, "sentence_len": 0, "sentence_num": 0}
  padded_shapes = (feature_shapes, [])
  padding_values = (pad_values, 0)
  dataset = dataset.padded_batch(params.batch_size, padded_shapes, padding_values).prefetch(1)
  print(dataset.output_types)
  print(dataset.output_shapes)
  return dataset

if __name__ == "__main__":
  params = {
	"max_sentence_len": 100,
	"max_sentence_num": 65,
	"pad_word": "<pad>",
	"batch_size": 1
  }
  dataset = input_fn("data/valid.csv", "data/words.txt", "environment_space", params, 0)
  iterator = dataset.make_initializable_iterator()
  next_element = iterator.get_next()
  with tf.Session() as sess:
    sess.run(iterator.initializer)
    print(sess.run(next_element))
    print(sess.run(next_element))
