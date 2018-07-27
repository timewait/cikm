# encoding:utf-8
import os
import random
import tensorflow as tf

from model import DANClassifier


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
    return {'inp': parsed['inp'], 'resp': parsed['label']}, parsed['label']


def train_input_fn(params):
    dataset = tf.data.TFRecordDataset(params.get('train_dirs'))
    dataset.map(parser).shuffle(buffer_size=params.get('buffer_size', 20000)).batch(
        params.get('batch_size', 256)).repeat(None)
    return dataset.make_one_shot_iterator()


def val_input_fn(params):
    dataset = tf.data.TFRecordDataset(params.get('val_dirs'))
    dataset.map(parser).shuffle(buffer_size=params.get('val_buffer_size', 10000)).batch(
        params.get('val_batch_size', 256)).repeat(params.get('epoch', 0))
    return dataset.make_one_shot_iterator()


class CIKMModel(DANClassifier):

    def embedding_layer(self, features, columns):
        inp_categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(key="inp", dtype=tf.int32,
                                                                                           vocabulary_file='',
                                                                                           default_value=0)
        inp_embedding_column = tf.feature_column.embedding_column(categorical_column=inp_categorical_column,
                                                                  dimension=8,
                                                                  combiner='sqrtn', trainable=True,
                                                                  initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                              stddev=0.02))
        resp_categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(key="resp", dtype=tf.int32,
                                                                                            vocabulary_file='',
                                                                                            default_value=0)
        resp_embedding_column = tf.feature_column.embedding_column(categorical_column=resp_categorical_column,
                                                                   dimension=8,
                                                                   combiner='sqrtn', trainable=True,
                                                                   initializer=tf.truncated_normal_initializer(mean=0.0,
                                                                                                               stddev=0.02))

        with tf.variable_scope("DAN", reuse=tf.AUTO_REUSE):
            inp_dense = tf.feature_column.input_layer(features, [inp_embedding_column])
            resp_dense = tf.feature_column.input_layer(features, [resp_embedding_column])
            inp_minus_resp = tf.abs(inp_dense - resp_dense)
            inp_mul_resp = inp_dense * resp_dense
            net = tf.layers.dense(tf.concat([inp_dense, resp_dense, inp_minus_resp, inp_mul_resp], axis=-1))
            return net


def main():
    params = {}
    model = CIKMModel().build(params)
    model.train(input_fn=train_input_fn, steps=params.get("steps", 1000))
    eval_result = model.eval(input_fn=val_input_fn)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    # predictions = model.predict(input_fn=train_input_fn, labels=None, batch_size=params.get("batch_size", 50))
    # print predictions


if __name__ == '__main__':
    main()
