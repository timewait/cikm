# encoding: utf-8
import tensorflow as tf
import abc


class DANClassifier(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def embedding_layer(self, features, columns=None):
        raise NotImplemented

    def model(self, features, labels, mode, params):
        with tf.variable_scope("dan", reuse=tf.AUTO_REUSE):
            embedding = self.embedding_layer(features, columns=params.get("features_columns", None))
            net = tf.reduce_mean(embedding, axis=1)
            for i, hidden in enumerate(params.get('hiddens', [512])):
                if hidden <= 0:
                    continue
            # TODO make activation, bias configurable
            net = tf.layers.dense(inputs=net, units=hidden, activation=tf.nn.tanh, use_bias=False, name="fc_%d" % i)

        logits = tf.squeeze(tf.layers.dense(net, params.get("n_classes", 1), name='logits'))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def build(self, params={}):
        return tf.estimator.Estimator(model_fn=self.mode, params=params)
