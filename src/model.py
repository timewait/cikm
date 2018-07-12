from abc import ABCMeta
from abc import abstractmethod

import logging
import tensorflow as tf
from .encoder import get_encoder_instance
import ops

logger = logging.getLogger('cikm')

def get_model_instance(hparams):
    kwargs = hparams.values()
    model_type = kwargs.get('model_type', 'dot_product')
    if model_type == 'dot_product':
        return DotProductModel(**kwargs)
    else:
        raise ValueError


class Model(object):
    """ Model.
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prefix = ''
        self.global_step = tf.train.get_or_create_global_step()
        self.max_gradient_norm = kwargs.get('max_gradient_norm', 5)
        self.starter_learning_rate = kwargs.get('learning_rate', 1e-3)
        self.decay_steps = kwargs.get('decay_steps', 100000)
        self.decay_rate = kwargs.get('decay_rate', 0.99)
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step, self.decay_steps, self.decay_rate, True, name='lr')
        self.optimizer_name = kwargs.get('optimizer', 'adam')
        self.activation_name = kwargs.get('activation', 'none')
        self.model_scope = kwargs.get('model_scope', 'model')
        self.local_init_op = tf.local_variables_initializer()
        self.activation_fn = ops.get_activation_fn(self.activation_name)
        self.eval_tf_op = {}
        self.eval_np_op = {}
        self.update_op = []
        self.metrics = {}
        self.summary = []

    def build(self, features, encoder=None, optimizer=None):
        """ Build.
        """
        logger.info('Start building %s ...' % self.__class__.__name__)
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = get_encoder_instance(**self.kwargs)
        logger.info('Start building encode ...')
        self._build_encode(features)

        with tf.variable_scope(self.model_scope):
            logger.info('Start building graph ...')
            self._build_graph()
            logger.info('Start building loss ...')
            self._build_loss()
            logger.info('Start building metrics ...')
            self._build_metrics()
            logger.info('Start building optimizer ...')
            self._build_optimizer(optimizer)
            logger.info('Start building summary ...')
            self._build_summary()
            logger.info('Start building eval_op ...')
            self._build_eval_op()

        logger.info('Finish building %s ...' % self.__class__.__name__)
        logger.info('global variables: ')
        for v in tf.global_variables():
            logger.info('var: %s  shape: %s' % (v.name, str(v.get_shape())))
        logger.info('trainable variables: ')
        for v in tf.trainable_variables():
            logger.info('var: %s  shape: %s' % (v.name, str(v.get_shape())))

    def build_infer(self, features, encoder=None):
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = get_encoder_instance(**self.kwargs)
        self._build_encode(features)
        with tf.variable_scope(self.model_scope):
            self._build_graph()
            self._build_loss()
            self._build_eval_op()

    @abstractmethod
    def _build_encode(self, features):
        pass

    @abstractmethod
    def _build_graph(self):
        pass

    @abstractmethod
    def _build_loss(self):
        pass

    @abstractmethod
    def _build_metrics(self):
        pass

    def _build_optimizer(self, optimizer=None):
        with tf.variable_scope('optimizer'):
            if optimizer is not None:
                self.optimizer = optimizer
            else:
                optimizer_fn = ops.get_optimizer_fn(self.optimizer_name)
                self.optimizer = optimizer_fn(self.learning_rate)
            gradients, trainable_vars = zip(*self.optimizer.compute_gradients(self.loss, tf.trainable_variables()))
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
            self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, trainable_vars), global_step=self.global_step)
        self.update_op.append(self.train_op)

    def _build_summary(self):
        for k, v in self.metrics.items():
            self.summary.append(tf.summary.scalar(k, v))
        self.merged_summary = tf.summary.merge(self.summary)

    @abstractmethod
    def _build_eval_op(self):
        pass

    def update(self, sess):
        sess.run(self.update_op)



class DotProductModel(Model):

    def __init__(self, **kwargs):
        super(DotProductModel, self).__init__(model_scope='dot_product', **kwargs)
        self.inp_dnn_sizes = kwargs.get('dot_inp_dnn_sizes', [300, 500])
        self.resp_dnn_sizes = kwargs.get('dot_resp_dnn_sizes', [300, 500])
        self.prefix = 'dot'

    def _build_encode(self, features):
        self.inp_enc = self.encoder.build(features['inp'])
        self.resp_enc = self.encoder.build(features['resp'])
        self.label = features['label']

    def _build_graph(self):
        inp_dnn = self._dnn(self.inp_enc, self.inp_dnn_sizes, 'inp_dnn')
        resp_dnn = self._dnn(self.resp_enc, self.resp_dnn_sizes, 'resp_dnn')
        inp_minus_resp = tf.abs(inp_dnn - resp_dnn)
        inp_mul_resp = inp_dnn * resp_dnn
        feat_concat = tf.concat([inp_dnn, resp_dnn, inp_minus_resp, inp_mul_resp], axis=-1)
        fc_out = tf.layers.dense(feat_concat, 512, name='fc')
        self.logits = tf.squeeze(tf.layers.dense(fc_out, 1, name='logits'))
        self.predictions = tf.where(self.logits > 0, tf.ones_like(self.logits), tf.zeros_like(self.logits))
        
    def _build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.label, tf.float32), logits=self.logits))

    def _build_metrics(self):
        with tf.variable_scope('train_metrics'):
            mean_loss, mean_loss_update = tf.metrics.mean(self.loss)
        self.update_op.append(mean_loss_update)
        self.metrics.update({'train_loss': mean_loss})

    def _build_summary(self):
        for k, v in self.metrics.items():
            self.summary.append(tf.summary.scalar(k, v))

        self.summary.append(tf.summary.histogram('inp_enc', self.inp_enc))
        self.summary.append(tf.summary.histogram('logits', self.logits))
        self.summary.append(tf.summary.scalar('learning_rate', self.learning_rate))
        self.merged_summary = tf.summary.merge(self.summary)

    def _build_eval_op(self):
        with tf.variable_scope('eval_metrics'):
            self.eval_tf_op = {'eval_loss': tf.metrics.mean(self.loss),
                               'eval_auc': tf.metrics.auc(labels=self.label, predictions=self.predictions),
                               'eval_acc': tf.metrics.accuracy(labels=self.label, predictions=self.predictions)}

    def _dnn(self, inputs, dnn_sizes, scope='dnn'):
        with tf.variable_scope(scope):
            for i, dnn_size in enumerate(dnn_sizes):
                if dnn_size <= 0:
                    continue
                inputs = tf.layers.dense(inputs=inputs, units=dnn_size, activation=self.activation_fn, name='fc%d' % i)
        return inputs

