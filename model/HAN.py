# coding=utf8
import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

class HAN(tf.estimator.Estimator):
  def __init__(self,
    params,
    model_dir=None,
    optimizer='Adagrad',
    config=None,
    warm_start_from=None,
  ):
    ''' an implement of Hierarchical Attention Networks for Document Classification '''
    if not optimizer: optimizer = 'Adagrad'
    self.optimizer = optimizers.get_optimizer_instance(optimizer, params.learning_rate)

    def _model_fn(features, labels, mode, params):
      # 构建模型
      word_embedded = self.word2vec(features["content"])
      sent_vec = self.sent2vec(word_embedded, features["sentence_len"], mode)
      doc_vec = self.doc2vec(sent_vec, features["sentence_num"], mode)
      is_training = mode == tf.estimator.ModeKeys.TRAIN
      if "doc_embedding_keep_rate" in params and params.doc_embedding_keep_rate < 1.0:
        doc_vec = tf.layers.dropout(doc_vec, params.doc_embedding_keep_rate, training=is_training)
      if params.num_classes == 2:
        my_head = tf.contrib.estimator.binary_classification_head()
      else:
        my_head = tf.contrib.estimator.multi_class_head(params.num_classes)
      logits = tf.layers.dense(doc_vec, my_head.logits_dimension, activation=None)
      return my_head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        logits=logits,
        train_op_fn=lambda loss: self.optimizer.minimize(loss, global_step=tf.train.get_global_step())
      )

    super(HAN, self).__init__(
      model_fn=_model_fn, model_dir=model_dir, config=config, params=params, warm_start_from=warm_start_from)

  def word2vec(self, x):
    # Get word embeddings for each token in the sentence
    embedding_mat = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[self._params.vocab_size, self._params.embedding_size])
    # shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
    word_embedded = tf.nn.embedding_lookup(embedding_mat, x)
    return word_embedded

  def sent2vec(self, word_embedded, word_num, mode):
    with tf.name_scope("sent2vec"):
      # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
      # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
      # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量
      # shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
      word_embedded = tf.reshape(word_embedded, [-1, self._params.max_sentence_len, self._params.embedding_size])
      word_num = tf.reshape(word_num, [-1])
      input_keep_rate = 1.0
      if mode == tf.estimator.ModeKeys.TRAIN and "word_embedding_keep_rate" in self._params:
        input_keep_rate = self._params["word_embedding_keep_rate"]
      # shape为[batch_size*sent_in_doce, word_in_sent, hidden_size*2]
      word_encoded = self.BidirectionalGRUEncoder(word_embedded, word_num, input_keep_rate, name='word_encoder')
      # shape为[batch_size*sent_in_doc, hidden_size*2]
      sent_vec = self.AttentionLayer(word_encoded, name='word_attention')
      return sent_vec

  def doc2vec(self, sent_vec, sent_num, mode):
    # 原理与sent2vec一样，根据文档中所有句子的向量构成一个文档向量
    with tf.name_scope("doc2vec"):
      sent_vec = tf.reshape(sent_vec, [-1, self._params.max_sentence_num, self._params.hidden_size * 2])
      input_keep_rate = 1.0
      if mode == tf.estimator.ModeKeys.TRAIN:
        if "sent_embedding_keep_rate" in self._params:
          input_keep_rate = self._params["sent_embedding_keep_rate"]
      # shape为[batch_size, sent_in_doc, hidden_size*2]
      doc_encoded = self.BidirectionalGRUEncoder(sent_vec, sent_num, input_keep_rate, name='sent_encoder')
      # shape为[batch_szie, hidden_szie*2]
      doc_vec = self.AttentionLayer(doc_encoded, name='sent_attention')
      return doc_vec

  def BidirectionalGRUEncoder(self, inputs, sequence_len, input_keep_rate, name):
    # 双向GRU的编码层，将一句话中的所有单词或者一个文档中的所有句子向量进行编码得到一个 2×hidden_size的输出向量，
    # 然后在经过Attention层，将所有的单词或句子的输出向量加权得到一个最终的句子/文档向量。
    # 输入inputs的shape是[batch_size, max_time, voc_size]
    with tf.variable_scope(name):
      cell_fw = tf.nn.rnn_cell.GRUCell(self._params.hidden_size)
      cell_bw = tf.nn.rnn_cell.GRUCell(self._params.hidden_size)
      if input_keep_rate < 1.0:
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_rate)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_rate)
      # fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
      ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                                           cell_bw=cell_bw,
                                                                           inputs=inputs,
                                                                           sequence_length=sequence_len,
                                                                           dtype=tf.float32)
      # outputs的size是[batch_size, max_time, hidden_size*2]
      outputs = tf.concat((fw_outputs, bw_outputs), 2)
      return outputs

  def AttentionLayer(self, inputs, name):
    # inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
    with tf.variable_scope(name):
      # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
      # 因为使用双向GRU，所以其长度为2×hidden_szie
      hidden_size = self._params.hidden_size * 2
      u_context = tf.get_variable('u_context', [hidden_size], dtype=tf.float32)
      # 使用一个全连接层编码GRU的输出的到期隐层表示,输出h的size是[batch_size, max_time, hidden_size * 2]
      h = tf.layers.dense(inputs, units=hidden_size, activation=tf.nn.tanh)
      # shape为[batch_size, max_time, 1]
      #alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
      alpha = tf.nn.softmax(tf.tensordot(h, u_context, [[-1], [0]])) # shape = [batch_size, max_time]
      alpha = tf.expand_dims(alpha, -1)
      # reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
      atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
      return atten_output
