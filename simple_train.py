import os
import sys
import logging
import argparse
import tensorflow as tf
import numpy as np
from src.dataset import get_data_iterator
from src.encoder import DANEncoder
from src import util

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

def train_and_eval(hparams):

    # define data
    train_dataset = get_dataset_from_dirs(hparams.train_dirs, hparams.train_batch_size, is_train=True)
    val_dataset = get_dataset_from_dirs(hparams.val_dirs, hparams.val_batch_size, is_train=False)
    iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    inp, resp, label = iterator.get_next()
    label = tf.cast(tf.squeeze(label), tf.float32)
    train_iter_init_op = iterator.make_initializer(train_dataset)
    val_iter_init_op = iterator.make_initializer(val_dataset)

    # define model
    encoder = DANEncoder(**hparams.values())
    inp_enc = encoder.build(inp)
    resp_enc = encoder.build(resp)
    inp_minus_resp = tf.abs(inp_enc - resp_enc)
    inp_mul_resp = inp_enc * resp_enc
    feat_concat = tf.concat([inp_enc, resp_enc, inp_minus_resp, inp_mul_resp], axis=-1)
    fc_out = tf.layers.dense(feat_concat, 512, name='fc')
    logits = tf.squeeze(tf.layers.dense(fc_out, 1, name='logits'))
    predictions = tf.where(logits > 0, x=tf.ones_like(logits), y=tf.zeros_like(logits))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits))
    train_op = tf.train.AdamOptimizer(hparams.learning_rate).minimize(loss)
    auc, auc_update_op = tf.metrics.auc(labels=label, predictions=predictions)
    acc, acc_update_op = tf.metrics.accuracy(labels=label, predictions=predictions)
    mean_loss, mean_loss_update_op = tf.metrics.mean(loss)
    saver = tf.train.Saver(max_to_keep=1)
    best_auc = 0.0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for i in xrange(1, 1000000):
            # train
            sess.run(train_iter_init_op)
            sess.run(tf.local_variables_initializer())
            for _ in xrange(500):
                sess.run([train_op, auc_update_op, acc_update_op, mean_loss_update_op])

            train_mean_loss, train_auc, train_acc = sess.run([mean_loss, auc, acc])
            print 'step %d:' % (i * 500)
            print '\ttrain:'
            print '\t\tloss: %f\tauc: %f\tacc: %f' % (train_mean_loss, train_auc, train_acc)


            # valid
            sess.run(val_iter_init_op)
            sess.run(tf.local_variables_initializer())
            while True:
                try:
                    sess.run([auc_update_op, acc_update_op, mean_loss_update_op])
                except tf.errors.OutOfRangeError:
                    break

            val_mean_loss, val_auc, val_acc = sess.run([mean_loss, auc, acc])
            print '\tval:'
            print '\t\tloss: %f\tauc: %f\tacc: %f' % (val_mean_loss, val_auc, val_acc)

            if val_auc > best_auc:
                best_auc = val_auc
                print 'save checkpoint ...'
                saver.save(sess, os.path.join(hparams.experiment_dir, 'model.ckpt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config path', default='config/maga.yaml')
    options = parser.parse_args(sys.argv[1:])

    hparams = util.load_hparams(options.config)
    os.system('rm -r %s' % (hparams.experiment_dir))
    os.system('mkdir -p %s' % (hparams.experiment_dir))
    os.system('cp %s %s' % (options.config, os.path.join(hparams.experiment_dir, 'config.yaml')))

    train_and_eval(hparams)

if __name__ == '__main__':
    main()
