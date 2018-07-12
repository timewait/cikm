import sys
import os
import tensorflow as tf
import numpy as np


def attention_bias_lower_triangle(length):
    band = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    band = tf.reshape(band, [1, 1, length, length])
    return -1e9 * (1.0 - band)

def attention_bias_by_pad_indicator(memory_padding):
    ret = memory_padding * -1e9
    return tf.expand_dims(tf.expand_dims(ret, axis=1), axis=1)

class MultiHeadAttention(object):
    def __init__(self, num_heads=1, linear_key_dim=50, linear_value_dim=50,
                 hidden_size=100, dropout=1.0, attention_bias=None):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.attention_bias = attention_bias


    def build(self, q, k, v):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.hidden_size, use_bias=False)

        return tf.nn.dropout(output, keep_prob=self.dropout)

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1: -1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3])  # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads
        o1 = tf.matmul(qs, ks, transpose_b=True) / (key_dim_per_head ** 0.5)
        o2 = o1 + self.attention_bias if self.attention_bias is not None else o1
        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)


    def _concat_heads(self, outputs):

        def transpose_then_concat_last_two_dimension(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3])
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimension(outputs)
