import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorflow.python.estimator.canned import optimizers

class TextCNN(tf.estimator.Estimator):
  def __init__(self, params, model_dir=None, config=None, optimizer='Adagrad', warm_start_from=None):
    if not optimizer: optimizer = 'Adagrad'
    self.optimizer = optimizers.get_optimizer_instance(optimizer, params.learning_rate)

    def _build_fully_connect_layers(net, hidden_units, dropout_rate, mode):
      # Build the hidden layers, sized according to the 'hidden_units' param.
      for units in hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if dropout_rate > 0.0:
          net = tf.layers.dropout(net, dropout_rate, training=(mode == tf.estimator.ModeKeys.TRAIN))
      return net

    def _get_f1_score_metric(labels, predictions, num_classes):
      ops = []
      f1_score = 0
      for i in range(num_classes):
        f1, precision, recall = _get_f1_score(labels, predictions, i)
        ops.append(precision)
        ops.append(recall)
        f1_score += f1
      f1_score /= num_classes
      return f1_score, tf.group(ops)

    def _get_f1_score(labels, predictions, n_class):
      pred = tf.equal(predictions, n_class)
      label = tf.equal(labels, n_class)
      precision = tf.metrics.precision(label, pred)
      recall = tf.metrics.recall(label, pred)
      f1_score = 2 * precision[0] * recall[0] / (precision[0] + recall[0] + 1e-5)
      return f1_score, precision, recall

    def _model_fn(features, labels, mode, config):
      sentence = features.pop('content')
      # Get word embeddings for each token in the sentence
      embeddings = tf.get_variable(name="embeddings", dtype=tf.float32,
                                   shape=[params.vocab_size, params.embedding_size])
      sentence = tf.nn.embedding_lookup(embeddings, sentence)  # shape:(batch, sentence_len, embedding_size)
      # add a channel dim, required by the conv2d and max_pooling2d method
      sentence = tf.expand_dims(sentence, -1)  # shape:(batch, sentence_len/height, embedding_size/width, channels=1)

      pooled_outputs = []
      for filter_size in params.filter_sizes:
        conv = tf.layers.conv2d(
          sentence,
          filters=params.num_filters,
          kernel_size=[filter_size, params.embedding_size],
          strides=(1, 1),
          padding="VALID",
          activation=tf.nn.relu)
        pool = tf.layers.max_pooling2d(
          conv,
          pool_size=[params.sentence_max_len - filter_size + 1, 1],
          strides=(1, 1),
          padding="VALID")
        pooled_outputs.append(pool)
      h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
      h_pool_flat = tf.reshape(h_pool, [-1, params.num_filters * len(params.filter_sizes)])  # shape: (batch, len(filter_size) * embedding_size)
      dropout_rate = params.dropout_rate if "dropout_rate" in params else 0.0
      last_common_layer = _build_fully_connect_layers(h_pool_flat, params.hidden_units, dropout_rate, mode)
      predictions = {"id": features.pop("id"), "content": tf.constant([""], dtype=tf.string)}
      metrics = {}
      loss = 0
      mean_f1_score = 0
      f1_dep_ops = []
      for k, v in features.iteritems():
        with tf.name_scope(k):
          net = _build_fully_connect_layers(last_common_layer, params.task_hidden_units, dropout_rate, mode)
          one_logits = tf.layers.dense(net, params.num_classes, activation=None, name=k + "_logits")
          predict_classes = tf.argmax(one_logits, 1)
          predictions[k] = predict_classes - 2
          if mode == tf.estimator.ModeKeys.PREDICT:
            continue
          cur_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=one_logits, labels=v))
          loss += cur_loss
          tf.summary.scalar("loss", cur_loss)
          acc_key = k + "/accuracy_1"
          metrics[acc_key] = tf.metrics.accuracy(labels=v, predictions=predict_classes)
          tf.summary.scalar("accuracy", metrics[acc_key][1])
          f1_score = _get_f1_score_metric(v, predict_classes, params.num_classes)
          metrics[k + "/f1_score"] = f1_score
          f1_dep_ops.append(f1_score[1])
          mean_f1_score += f1_score[0]

      if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = { 'prediction': tf.estimator.export.PredictOutput(predictions) }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

      if mode == tf.estimator.ModeKeys.EVAL:
        metrics["f1_score"] = (mean_f1_score, tf.group(f1_dep_ops))
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

      assert mode == tf.estimator.ModeKeys.TRAIN
      train_op = self.optimizer.minimize(loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    super(TextCNN, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config,
        warm_start_from=warm_start_from)
