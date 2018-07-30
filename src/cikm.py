# encoding:utf-8
import os
import random
import tensorflow as tf

from models.DANClassifier import DANClassifier


# 1. input, output
def get_paths_in_dirs(dirs, shuffle=False):
    paths = []
    for dir in dirs:
        paths.extend(list(map(lambda filename: os.path.join(dir, filename), os.listdir(dir))))
    if shuffle is True:
        random.shuffle(paths)
    return paths


def get_dataset_from_dirs(data_dirs, batch_size, is_train=True):
    def parse_exmp(serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={'inp': tf.VarLenFeature(tf.int64),
                                                               'resp': tf.VarLenFeature(tf.int64),
                                                               'label': tf.FixedLenFeature([1], tf.int64)})
        inp = tf.sparse_tensor_to_dense(feats['inp'])[:100]
        resp = tf.sparse_tensor_to_dense(feats['resp'])[:100]
        label = feats['label']
        return inp, resp, label

    paths = get_paths_in_dirs(data_dirs)
    dataset = tf.data.TFRecordDataset(paths, 'GZIP').map(parse_exmp)
    padded_shapes = tuple(dataset.output_shapes)
    padded_values = tuple([tf.convert_to_tensor(0, tf.int64)] * len(padded_shapes))
    if is_train is True:
        dataset = dataset.shuffle(buffer_size=20000).repeat(None)
    dataset = dataset.padded_batch(batch_size, padded_shapes, padded_values).prefetch(10)
    return dataset


def parser(record):
    parsed = tf.parse_single_example(record, features={
        'inp': tf.VarLenFeature(tf.int64),
        'resp': tf.VarLenFeature(tf.int64),
        'label': tf.FixedLenFeature([1], tf.int64)
    })
    return {'inp': parsed['inp'], 'resp': parsed['label']}, tf.cast(tf.squeeze(parsed['label']), dtype=tf.float32)


def train_input_fn(params):
    dataset = tf.data.TFRecordDataset(params.get('train_dirs'), 'GZIP')
    dataset = dataset.map(parser).shuffle(buffer_size=params.get('buffer_size', 20000)).batch(
        params.get('batch_size', 256)).repeat(None)
    return dataset


def val_input_fn(params):
    dataset = tf.data.TFRecordDataset(params.get('val_dirs'), 'GZIP')
    dataset = dataset.map(parser).shuffle(buffer_size=params.get('buffer_size', 20000)).batch(
        params.get('batch_size', 256)).repeat(10)
    return dataset


class CIKMModel(DANClassifier):

    def __init__(self):
        super(CIKMModel, self).__init__()

    def embedding_dense(self, features, column, vocab_file):
        with tf.name_scope('embedding_%s' % column):
            categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(key=column, dtype=tf.int32,
                                                                                           vocabulary_file=vocab_file,
                                                                                           default_value=0)

            embedding_column = tf.feature_column.embedding_column(categorical_column=categorical_column,
                                                                  dimension=256,
                                                                  combiner='sqrtn', trainable=True,
                                                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                              stddev=0.02))
            input_layer = tf.feature_column.input_layer(features, [embedding_column])
        with tf.name_scope('average_%s' % column):
            net = tf.reduce_mean(input_layer, axis=1, keep_dims=True)
        for i, hidden_size in enumerate([512]):
            if hidden_size <= 0:
                continue
            net = tf.layers.dense(inputs=net, units=hidden_size, activation=tf.nn.tanh, use_bias=True,
                                  name='fc_%d' % i)
        return net

    def layers(self, features, vocab_file):
        with tf.variable_scope('input'):
            inp_dense = self.embedding_dense(features, 'inp', vocab_file)
            resp_dense = self.embedding_dense(features, 'resp', vocab_file)

        with tf.variable_scope("dan", reuse=tf.AUTO_REUSE):
            inp_minus_resp = tf.abs(inp_dense - resp_dense)
            inp_mul_resp = inp_dense * resp_dense
            net = tf.layers.dense(tf.concat([inp_dense, resp_dense, inp_minus_resp, inp_mul_resp], axis=-1), units=512)
            return net


def main():
    params = {
        'train_dirs': ['/Users/michell/AnacondaProjects/koubei/cikm/data/tfrecord/train/cikm_en_train.tfrecord'],
        'val_dirs': ['/Users/michell/AnacondaProjects/koubei/cikm/data/tfrecord/val/cikm_sp_val.tfrecord'],
        'vocab_file': '/Users/michell/AnacondaProjects/koubei/cikm/data/dict/vocab_ids.txt'
    }
    model = CIKMModel().build(params)
    for i in range(100):
        model.train(input_fn=lambda: train_input_fn(params), steps=params.get("steps", 20))
        eval_result = model.evaluate(input_fn=lambda: val_input_fn(params))
        print(eval_result)
    # predictions = model.predict(input_fn=train_input_fn, labels=None, batch_size=params.get("batch_size", 50))
    # print predictions


if __name__ == '__main__':
    main()
