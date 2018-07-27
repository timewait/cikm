# encoding: utf-8
import tensorflow as tf
import abc


class DANClassifier(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def layers(self, features, vocab_file):
        raise NotImplemented

    def model(self, features, labels, mode, params):
        with tf.variable_scope("dan", reuse=tf.AUTO_REUSE):
            net = self.layers(features, vocab_file=params.get("vocab_file"))

        logits = tf.squeeze(tf.layers.dense(net, params.get("n_classes", 1), name='logits'))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        predictions = tf.where(logits > 0, x=tf.ones_like(logits), y=tf.zeros_like(logits))

        if mode == tf.estimator.ModeKeys.PREDICT:
            prediction = {
                'class_ids': predictions[:, tf.newaxis],
                'probabilities': tf.nn.softmax(logits),
                'logits': logits
            }
            return tf.estimator.EstimatorSpec(mode, predictions=prediction)

        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc_op')
        auc = tf.metrics.auc(labels, predictions=predictions, name='auc_op')
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('auc', auc[1])

        metrics = {
            'accuracy': accuracy,
            'auc': auc
        }

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics
            )

        assert mode == tf.estimator.ModeKeys.TRAIN
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def build(self, params={}):
        return tf.estimator.Estimator(model_fn=self.model, params=params)
