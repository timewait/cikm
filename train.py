import os
import sys
import logging
import argparse
import tensorflow as tf
import numpy as np
from src.model import get_model_instance
from src.dataset import get_data_iterator
from src import util
from colorlog import ColoredFormatter

def init_env(hparams):

    formatter = ColoredFormatter(
        fmt='%(log_color)s[%(levelname)s] %(asctime)s [%(filename)12s:%(lineno)5d]:\t%(message)s%(reset)s')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(hparams.experiment_dir, 'log.txt'))
    file_handler.setFormatter(formatter)
    logger = logging.getLogger('cikm')
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    tf.logging.set_verbosity(tf.logging.DEBUG)

    tf.set_random_seed(321)
    np.random.seed(123)


def train_and_eval(hparams):
    logger = logging.getLogger('cikm')

    with tf.Graph().as_default() as train_graph:
        iterator = get_data_iterator(hparams, is_train=True)
        train_model = get_model_instance(hparams)
        train_model.build(iterator.features)
        train_config = tf.ConfigProto()
        train_config.gpu_options.allow_growth = True

        saver = tf.train.Saver(max_to_keep=1)
        scaffold = tf.train.Scaffold(saver=saver)
        session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold,
                                                       checkpoint_dir=hparams.experiment_dir,
                                                       master='',
                                                       config=train_config)

        hooks = [
            tf.train.StopAtStepHook(last_step=10 * hparams.max_epoch * hparams.steps_per_checkpoint),
            tf.train.StepCounterHook(output_dir=hparams.experiment_dir,
                                     every_n_steps=hparams.steps_per_info),
            tf.train.CheckpointSaverHook(hparams.experiment_dir,
                                         scaffold=scaffold,
                                         save_steps=hparams.steps_per_checkpoint),
            util.IteratorInitializerHook(iterator=iterator),
            util.TrainSummarySaverHook(save_steps=hparams.steps_per_info,
                                       output_dir=hparams.experiment_dir,
                                       scaffold=scaffold,
                                       init_op=train_model.local_init_op),
            util.EvalHook(hparams=hparams),
        ]

        if hparams.debug is True:
            vocab = util.load_vocab('data/dict/vocab.txt')
            debug_data = {k: v for k, v in iterator.features.items() if 'label' not in k}
            hooks.extend([
                util.DebugDataHook(data=debug_data,
                                   vocab=vocab,
                                   steps_per_debug=50),
            ])

        with tf.train.MonitoredSession(session_creator=session_creator, hooks=hooks) as sess:
            while not sess.should_stop():
                train_model.update(sess)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config path', default='config/maga.yaml')
    options = parser.parse_args(sys.argv[1:])

    hparams = util.load_hparams(options.config)
    os.system('mkdir -p %s' % (hparams.experiment_dir))
    os.system('cp %s %s' % (options.config, os.path.join(hparams.experiment_dir, 'config.yaml')))

    init_env(hparams)
    train_and_eval(hparams)

if __name__ == '__main__':
    main()

