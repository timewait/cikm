import os
import random
import tensorflow as tf
import numpy as np
from abc import ABCMeta
from abc import abstractmethod


def get_data_iterator(hparams, is_train=True):
    kwargs = hparams.values()
    model_type = kwargs.get('model_type', 'dot_product')
    encoder_type = kwargs.get('encoder_type', 'dan')
    max_seqlen = kwargs.get('max_seqlen', 200)
    fix_seqlen = True if encoder_type == 'transformer' else False
    if is_train is True:
        batch_size = kwargs.get('train_batch_size', 50)
    else:
        batch_size = kwargs.get('val_batch_size', 50)
    if model_type == 'dot_product':
        if is_train is True:
            data_dirs = kwargs['train_dirs']
        else:
            data_dirs = kwargs['val_dirs']
        iterator = DotProductIterator(data_dirs, batch_size, max_seqlen, fix_seqlen, is_train)
    else:
        raise ValueError

    return iterator

def get_paths_in_dirs(dirs, shuffle=False):
    paths = []
    for dir in dirs:
        paths.extend(list(map(lambda filename: os.path.join(dir, filename), os.listdir(dir))))
    if shuffle is True:
        random.shuffle(paths)
    return paths

class Iterator(object):
    __metaclass__ = ABCMeta

    def __init__(self, batch_size=1, max_seqlen=200, fix_seqlen=False, is_train=True, prefix=''):
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.fix_seqlen = fix_seqlen
        self.is_train = is_train
        self.prefix = prefix
        self.compression_type = 'GZIP'
        self.is_shuffle_paths = True
        self._dataset = None
        self._iterator = None
        self._features = {}

    def _default_pipeline(self, data_dirs):
        paths = get_paths_in_dirs(data_dirs, self.is_shuffle_paths)
        self._dataset = tf.data.TFRecordDataset(paths, self.compression_type).map(self._parse_exmp)
        if self.fix_seqlen is True:
            padded_shapes = tuple([tf.TensorShape([tf.Dimension(x_dim) if x_dim != None else tf.Dimension(self.max_seqlen)]) for x_shape in self._dataset.output_shapes for x_dim   in x_shape.as_list()])
        else:
            padded_shapes = tuple(self._dataset.output_shapes)
        padded_values = tuple([tf.convert_to_tensor(0, tf.int64)] * len(padded_shapes))
        if self.is_train is True:
            self._dataset = self._dataset.shuffle(buffer_size=self.batch_size * 100).repeat(None)
        self._dataset = self._dataset.padded_batch(self.batch_size, padded_shapes, padded_values).prefetch(10)
        self._iterator = self._dataset.make_initializable_iterator()
        return self._iterator.get_next()

    @abstractmethod
    def _parse_exmp(self, serial_exmp):
        pass

    @property
    def features(self):
        return self._features

    @property
    def initializer(self):
        return self._iterator.initializer


class DotProductIterator(Iterator):
    def __init__(self, data_dirs, batch_size=1, max_seqlen=200, fix_seqlen=False, is_train=True, prefix='dot'):
        super(DotProductIterator, self).__init__(batch_size, max_seqlen, fix_seqlen, is_train, prefix)
        inp, resp, label = self._default_pipeline(data_dirs)
        self._features = {'inp': inp,
                          'resp': resp,
                          'label': label}

    def _parse_exmp(self, serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={'inp': tf.VarLenFeature(tf.int64),
                                                               'resp': tf.VarLenFeature(tf.int64),
                                                               'label': tf.FixedLenFeature([1], tf.int64)})
        inp = tf.sparse_tensor_to_dense(feats['inp'])[:self.max_seqlen]
        resp = tf.sparse_tensor_to_dense(feats['resp'])[:self.max_seqlen]
        label = feats['label']
        return inp, resp, label

