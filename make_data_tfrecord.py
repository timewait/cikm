import sys
import time
import os
import gzip
from pathos.multiprocessing import ProcessingPool as Pool
import tensorflow as tf
from collections import OrderedDict


def load_vocab(path, encoding='utf8'):
    vocab = {}
    vocab['idx2word'] = {}
    vocab['word2idx'] = {}
    with open(path, 'r') as f:
        for line in f:
            cols = line.decode(encoding, 'ignore').split('\t')
            word = cols[0]
            idx = len(vocab['word2idx'])
            vocab['word2idx'][word] = idx
            vocab['idx2word'][idx] = word
    print 'vocab size: %d' % (len(vocab['word2idx']))
    return vocab


def text_preprocess(text, encoding):
    return text.lower().decode(encoding, 'ignore')


def sentence_to_ids(sentence, vocab, seqlen, skip=False, with_pad=False):
    assert type(sentence) == list
    global debug_count
    global is_debug

    def pad(sentence):
        # pad id must be 0
        if len(sentence) >= seqlen:
            return sentence[:seqlen]
        else:
            return sentence + (seqlen - len(sentence)) * [0]

    word2idx = vocab['word2idx']
    sentence_ori = sentence
    sentence_ids = [word2idx[w] for w in sentence_ori if w in vocab['word2idx']]
    if is_debug is True and debug_count > 0:
        debug_count -= 1
        idx2word = vocab['idx2word']
        sentence_proc = [idx2word[w] for w in sentence_ids]
        print 'ori(%d): %s' % (len(sentence_ori), ' '.join(sentence_ori).encode('utf8'))
        print 'proc(%d): %s' % (len(sentence_proc), ' '.join(sentence_proc).encode('utf8'))
        print 'ids(%d): %s' % (len(sentence_ids), ' '.join(map(str, sentence_ids)))
        print

    if skip is True:
        if len(sentence_ids) > seqlen or len(sentence_ids) == 0:
            return None
    if with_pad is True:
        sentence_ids = pad(sentence_ids)
    return sentence_ids[:seqlen]


def loop_files_in_dir(input_dir, output_dir, loop_fn):
    def replace_postfix(x):
        if x.endswith('.txt'):
            x = x[:-4]
        elif x.endswith('.gz'):
            x = x[:-3]
        else:
            pass
        x = x + '.tfrecord'
        return x

    in_filenames = os.listdir(input_dir)
    out_filenames = map(replace_postfix, in_filenames)
    input_paths = map(lambda x: os.path.join(input_dir, x), in_filenames)
    output_paths = map(lambda x: os.path.join(output_dir, x), out_filenames)
    paths = list(zip(input_paths, output_paths))
    N = min(10, len(paths))

    global is_debug
    if is_debug is True:
        for in_path, out_path in paths:
            loop_fn((in_path, out_path))
    else:
        pool = Pool(N)
        pool.map(loop_fn, paths)


def make_dot_product_tfrecord(input_dir, output_dir, vocab, seqlen, sep='\t', encoding='utf8'):
    def get_dot_product_example(inp, resp, label):
        record_features = {}
        record_features['inp'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(inp)))
        record_features['resp'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(resp)))
        record_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(label)))
        return tf.train.Example(features=tf.train.Features(feature=record_features))

    def create_tfrecord(args):
        input_path, output_path = args
        option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        tfrecord_wrt = tf.python_io.TFRecordWriter(output_path, option)
        if input_path.endswith('.gz'):
            open_fn = gzip.open
        else:
            open_fn = open
        with open_fn(input_path, 'r') as f:
            for line in f:
                line = text_preprocess(line, encoding)
                cols = line.strip('\n').split(sep)
                if len(cols) != 3:
                    continue
                x, y, label = cols
                inp = sentence_to_ids(x.split(' '), vocab, seqlen)
                resp = sentence_to_ids(y.split(' '), vocab, seqlen)
                if inp is None or resp is None or len(inp) == 0 or len(resp) == 0:
                    continue
                exmp = get_dot_product_example(inp, resp, [int(label)])
                exmp_serial = exmp.SerializeToString()
                tfrecord_wrt.write(exmp_serial)
        tfrecord_wrt.close()

    loop_files_in_dir(input_dir, output_dir, create_tfrecord)


if __name__ == '__main__':
    global is_debug
    global debug_count
    is_debug = True
    debug_count = 10

    input_dir = 'data/text/val'
    output_dir = 'data/tfrecord/val'
    vocab = load_vocab('data/dict/vocab.txt', 'utf8')
    seqlen = 100
    os.system('mkdir -p %s' % (output_dir))

    make_dot_product_tfrecord(
        input_dir=input_dir,
        output_dir=output_dir,
        vocab=vocab,
        seqlen=seqlen,
    )
