import sys
import os
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
import attention
import ops


__all__ = ['get_encoder_instance', 'TransformerEncoder', 'DANEncoder']

def get_encoder_instance(**kwargs):
    encoder_type = kwargs.get('encoder_type', 'dan')
    if encoder_type == 'dan':
        return DANEncoder(**kwargs)
    elif encoder_type == 'transformer':
        return TransformerEncoder(**kwargs)
    else:
        raise ValueError

def get_embedding_variable_with_pad(vocab_size, embed_dim, initializer=tf.truncated_normal_initializer()):
    # make sure your pad id is 0
    pad_embedding = tf.zeros([1, embed_dim], dtype=tf.float32)
    word_embedding = tf.get_variable('embedding', shape=[vocab_size-1, embed_dim], dtype=tf.float32, initializer=initializer)
    embedding = tf.concat([pad_embedding, word_embedding], axis=0)
    return embedding

def pad_indicator_in_embedding(emb):
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.to_float(tf.equal(emb_sum, 0.0))


class PadRemover(object):
    def __init__(self, pad_indicator):
        self.nonpad_indicator = None
        self.dim_origim = None

        with tf.name_scope('pad_reduce/get_ids'):
            pad_indicator = tf.reshape(pad_indicator, [-1])
            self.nonpad_indicator = tf.to_int32(tf.where(pad_indicator < 1e-9))
            self.dim_origin = tf.shape(pad_indicator)[:1]

    def remove(self, x):
        with tf.name_scope('pad_reduce/remove'):
            x_shape = x.get_shape().as_list()
            x = tf.gather_nd(x, indices=self.nonpad_indicator)
        return x

    def restore(self, x):
        with tf.name_scope('pad_reduce/restore'):
            x = tf.scatter_nd(indices=self.nonpad_indicator, updates=x, shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0))
        return x

class Encoder(object):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.embed_dim = kwargs.get('embed_dim', 128)
        self.vocab_size = kwargs.get('vocab_size', 1500000)
        self.activation_name = kwargs.get('activation', 'none')
        self.initializer_fn = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
        self.activation_fn = ops.get_activation_fn(self.activation_name)
        self.nodes = {}

    def build(self, inputs, reuse=True):
        with tf.variable_scope(self.local_scope, reuse=tf.AUTO_REUSE):
            embedding = get_embedding_variable_with_pad(self.vocab_size, self.embed_dim, self.initializer_fn)
            inputs_emb = tf.nn.embedding_lookup(embedding, inputs)
            output = self._build_graph(inputs_emb)
        return output

    @abstractmethod
    def _build_graph(self, inputs):
        pass

    @property
    def variables(self):
        return tf.trainable_variables(self.local_scope)

    def build_infer_encode(self, sent):
        self.sent_enc = self.build(sent, reuse=False)

    def build_infer_matrix_cosine(self, sent):
        self.sent_enc = self.build(sent, reuse=False)
        self.matrix_cosine = ops.cosine_similarity(self.sent_enc, self.sent_enc, mode='matrix')

    def build_infer_vector_cosine(self, sent1, sent2):
        self.sent1_enc = self.build(sent1, reuse=False)
        self.sent2_enc = self.build(sent2, reuse=True)
        self.vector_cosine = ops.cosine_similarity(self.sent1_enc, self.sent2_enc, mode='vector')


class TransformerEncoder(Encoder):
    def __init__(self, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.local_scope = 'transformer_encoder'
        self.num_layer = kwargs.get('T_num_layer', 1)
        self.num_heads = kwargs.get('T_num_heads', 1)
        self.linear_key_dim = kwargs.get('T_linear_key_dim', 50)
        self.linear_value_dim = kwargs.get('T_linear_value_dim', 50)
        self.kernel_size = kwargs.get('T_kernel_size', 50)
        self.hidden_size = kwargs.get('T_hidden_size', 50)
        self.dropout = kwargs.get('T_dropout', 1.0)
        self.use_positional_encoding = kwargs.get('T_use_positional_encoding', True)
        self.use_pad_remover = kwargs.get('T_use_pad_remover', True)

    def _build_graph(self, inputs_emb):
        pad_indicator = pad_indicator_in_embedding(inputs_emb)
        attention_bias = attention.attention_bias_by_pad_indicator(pad_indicator)
        pad_remover = None
        if self.use_pad_remover is True:
            pad_remover = PadRemover(pad_indicator)
        if self.use_positional_encoding is True:
            batch_size = tf.shape(inputs_emb)[0]
            seqlen, dim = inputs_emb.get_shape().as_list()[1:]
            position_inputs = self._positional_encoding(batch_size, seqlen, dim)
            encoded_inputs = tf.add(inputs_emb, position_inputs)
        else:
            encoded_inputs = inputs_emb
        o1 = tf.identity(encoded_inputs)
        for i in range(self.num_layer):
            with tf.variable_scope('layer-%d' % i):
                o2 = self._layer_postprocess(o1,
                                             self._self_attention(self._layer_preprocess(o1, num=1), attention_bias),
                                             num=1)
                o3 = self._layer_postprocess(o2,
                                             self._position_wise_feed_forward(self._layer_preprocess(o2, num=2), pad_remover),
                                             num=2)
                o1 = tf.identity(o3)
        with tf.variable_scope('layer-out'):
            nonpad_cnt = tf.reduce_sum(pad_indicator, axis=1, keep_dims=True)
            ignore = tf.ones_like(nonpad_cnt, dtype=tf.float32) * 1e9
            nonpad_cnt = tf.where(nonpad_cnt < 1e-9, ignore, nonpad_cnt)
            output = tf.reduce_sum(o3, 1) / nonpad_cnt
            output = tf.layers.dense(output, self.hidden_size)
        return output

    def _positional_encoding(self, batch_size, seqlen, dim):

        with tf.variable_scope('positional-encoding'):
            encoded_vec = np.array([pos/np.power(10000, 2*(i//2)/dim) for pos in range(seqlen) for i in range(dim)])
            encoded_vec[::2] = np.sin(encoded_vec[::2])
            encoded_vec[1::2] = np.cos(encoded_vec[1::2])
            positional_embedding = tf.convert_to_tensor(encoded_vec.reshape([seqlen, dim]), dtype=tf.float32)
            position_inputs = tf.reshape(tf.tile(tf.range(0, seqlen), [batch_size]), [batch_size, seqlen])
            position_inputs = tf.nn.embedding_lookup(positional_embedding, position_inputs)
        return position_inputs

    def _self_attention(self, x, attention_bias=None):
        with tf.variable_scope('self-attention'):
            multi_head_attention = attention.MultiHeadAttention(num_heads=self.num_heads,
                                                                linear_key_dim=self.linear_key_dim,
                                                                linear_value_dim=self.linear_value_dim,
                                                                hidden_size=self.hidden_size,
                                                                dropout=self.dropout,
                                                                attention_bias=attention_bias)
            return multi_head_attention.build(x, x, x)

    def _layer_preprocess(self, x, num=0):
        with tf.variable_scope('norm-%d' % num):
            return tf.contrib.layers.layer_norm(x)

    def _layer_postprocess(self, x, y, num=0):
        with tf.variable_scope('dropout-residual-%d' % num):
            y = tf.nn.dropout(y, keep_prob=self.dropout)
            y += x
        return y

    def _add_and_norm(self, x, sub_layer_x, num=0):
        with tf.variable_scope('add-and-norm-%d' % num):
            return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x))

    def _position_wise_feed_forward(self, input, pad_remover=None):

        if pad_remover is not None:
            input_shape = tf.shape(input)
            input = tf.reshape(input, tf.concat([[-1], input_shape[2:]], axis=0))
            input = tf.expand_dims(pad_remover.remove(input), axis=0)

        with tf.variable_scope('position-wise-feed-forward'):
            output = tf.layers.dense(input, self.kernel_size, activation=tf.nn.relu)
            output = tf.layers.dense(output, self.hidden_size)

        if pad_remover is not None:
            output = tf.reshape(pad_remover.restore(tf.squeeze(output, axis=0)), input_shape)

        return output


class DANEncoder(Encoder):

    def __init__(self, **kwargs):
        super(DANEncoder, self).__init__(**kwargs)
        self.local_scope = 'dan'
        self.hidden_sizes = kwargs.get('D_hidden_sizes', [300, 300, 500])

    def _build_graph(self, inputs_emb):
        pad_indicator = pad_indicator_in_embedding(inputs_emb)
        nonpad_cnt = tf.maximum(tf.reduce_sum(pad_indicator, axis=1, keep_dims=True), 1e-9)
        inputs_avg = tf.reduce_sum(inputs_emb, 1) / tf.sqrt(nonpad_cnt)
        inp = inputs_avg
        for i, hidden_size in enumerate(self.hidden_sizes):
            if hidden_size <= 0:
                continue
            inp = tf.layers.dense(inputs=inp, units=hidden_size, activation=self.activation_fn, use_bias=False, name='fc%d' % i)
        return inp
