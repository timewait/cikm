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


def get_embedding_variable_with_pad(vocab_size, embed_dim, initializer=tf.truncated_normal_initializer()):
    # make sure your pad id is 0
    pad_embedding = tf.zeros([1, embed_dim], dtype=tf.float32)
    word_embedding = tf.get_variable('embedding', shape=[vocab_size - 1, embed_dim], dtype=tf.float32,
                                     initializer=initializer)
    embedding = tf.concat([pad_embedding, word_embedding], axis=0)
    return embedding


def pad_indicator_in_embedding(emb):
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.to_float(tf.equal(emb_sum, 0.0))


def input_fn(hparams):
    # define data
    train_dataset = get_dataset_from_dirs(hparams.train_dirs, hparams.train_batch_size, is_train=True)
    val_dataset = get_dataset_from_dirs(hparams.val_dirs, hparams.val_batch_size, is_train=False)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    inp, resp, label = iterator.get_next()
    label = tf.cast(tf.squeeze(label), tf.float32)

    train_iter_init_op = iterator.make_initializer(train_dataset)
    val_iter_init_op = iterator.make_initializer(val_dataset)

    return train_iter_init_op, val_iter_init_op


class CIKMModel(DANClassifier):

    def embedding_layer(self, features, columns):
        pass


def main():
    params = {}
    model = CIKMModel().build(params)
    model.train(input_fn=input_fn, steps=params.get("steps", 1000))
    eval_result = model.eval(input_fn=input_fn)
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    predictions = model.predict(input_fn=input_fn, labels=None, batch_size=params.get("batch_size", 50))
    print predictions


if __name__ == '__main__':
    main()
