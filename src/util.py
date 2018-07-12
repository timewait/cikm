import os
import sys
import time
import logging
import numpy as np
import tensorflow as tf
from .dataset import get_data_iterator
from .model import get_model_instance
from .encoder import get_encoder_instance

logger = logging.getLogger('use')

def load_hparams(path):

    def parse_json(path):
        import json
        return json.loads(open(path, 'r'))

    def parse_yaml(path):
        import yaml
        return yaml.load(open(path, 'r'))

    fname, fextension = os.path.splitext(path)
    if fextension == '.json':
        config = parse_json(path)
    elif fextension == '.yaml':
        config = parse_yaml(path)
    else:
        raise ValueError

    hparams = tf.contrib.training.HParams(**config)
    return hparams

def load_vocab(path, encoding='utf8'):
    vocab = {}
    vocab['idx2word'] = {}
    vocab['word2idx'] = {}
    with open(path, 'r') as f:
        for line in f:
            cols = line.decode(encoding).split('\t')
            word = cols[0]
            idx = len(vocab['word2idx'])
            vocab['word2idx'][word] = idx
            vocab['idx2word'][idx] = word
    return vocab

class IteratorInitializerHook(tf.train.SessionRunHook):
    def __init__(self, iterator):
        super(IteratorInitializerHook, self).__init__()
        self._iterator = iterator

    def after_create_session(self, session, coord):
        session.run(self._iterator.initializer)


class EvalHook(tf.train.SessionRunHook):

    def __init__(self, hparams):
        self.hparams = hparams

    def begin(self):
        logger.info('Start creating EvalHook ...')
        self._global_step = tf.train.get_global_step()
        self.eval_writer = tf.summary.FileWriterCache.get(self.hparams.experiment_dir)
        with tf.Graph().as_default() as self.eval_graph:
            self.iterator = get_data_iterator(self.hparams, is_train=False)
            self.eval_model = get_model_instance(self.hparams)
            self.eval_model.build_infer(self.iterator.features)
            self.eval_config = tf.ConfigProto(device_count={'GPU': 0})
            self.eval_session_creator=tf.train.ChiefSessionCreator(scaffold=tf.train.Scaffold(),
                                                                   checkpoint_dir=self.hparams.experiment_dir, config=self.eval_config)
            self.eval_hooks = [IteratorInitializerHook(iterator=self.iterator)]
            self.eval_tf_metrics = {k: v[0] for k, v in self.eval_model.eval_tf_op.items()}
            self.eval_tf_updates = {k: v[1] for k, v in self.eval_model.eval_tf_op.items()}
            self.eval_np_metrics = {k: v for k, v in self.eval_model.eval_np_op.items()}
            self.eval_np_outputs = {k: v.sess_run_op for k, v in self.eval_model.eval_np_op.items()}

            logger.info('eval tf metrics:')
            for k in self.eval_tf_metrics.keys():
                logger.info('\t%s' % k)
            logger.info('eval np metrics:')
            for k in self.eval_np_metrics.keys():
                logger.info('\t%s' % k)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step)

    def after_run(self, run_context, run_values):
        step_value = int(run_values.results)
        if step_value % self.hparams.steps_per_eval == 0 and step_value >= self.hparams.steps_per_eval:
            logger.info('eval validation set')
            with self.eval_graph.as_default():
                with tf.train.MonitoredSession(session_creator=self.eval_session_creator, hooks=self.eval_hooks) as eval_sess:
                    tic = time.time()
                    for k, v in self.eval_np_metrics.items():
                        v.reset()

                    try:
                        while True:
                            _, np_outputs = eval_sess.run([self.eval_tf_updates, self.eval_np_outputs])
                            for k, v in np_outputs.items():
                                labels, scores = v
                                self.eval_np_metrics[k].add_and_update(labels, scores)
                    except tf.errors.OutOfRangeError:
                        pass
                    tf_metrics = eval_sess.run(self.eval_tf_metrics)
                    proto = [tf.Summary.Value(tag=k, simple_value=v) for k, v in tf_metrics.items()]
                    proto.extend([tf.Summary.Value(tag=k, simple_value=v.value()) for k, v in self.eval_np_metrics.items()])
                    self.eval_writer.add_summary(summary=tf.Summary(value=proto), global_step=step_value)
                    self.eval_writer.flush()
                    tac = time.time()
                    for v in proto:
                        logger.info('\t%s: %f' % (v.tag, v.simple_value))
                    logger.info('time consume: %f sec' % (tac - tic))

class TrainSummarySaverHook(tf.train.SessionRunHook):
    def __init__(self, save_steps=None, save_secs=None, output_dir=None, scaffold=None, summary_op=None, init_op=None):
        if ((scaffold is None and summary_op is None) or (scaffold is not None and summary_op is not None)):
            raise ValueError('Exactly one of scaffold or summary_op must be provided')

        self._summary_op = summary_op
        self._output_dir = output_dir
        self._scaffold = scaffold
        self._init_op = init_op
        self._timer = tf.train.SecondOrStepTimer(every_secs=save_secs,
                                                 every_steps=save_steps)

    def begin(self):
        self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
        self._next_step = None
        self._global_step_tensor = tf.train.get_or_create_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use SummarySaverHook')

    def before_run(self, run_context):
        self._request_summary = (self._next_step is None or self._timer.should_trigger_for_step(self._next_step))
        requests = {'global_step': self._global_step_tensor}
        if self._request_summary and self._next_step is not None:
            if self._get_summary_op() is not None:
                requests['summary'] = self._get_summary_op()
                run_context.session.run(self._init_op)
        return tf.train.SessionRunArgs(requests)


    def after_run(self, run_context, run_values):
        if not self._summary_writer:
            return
        stale_global_step = run_values.results['global_step']
        global_step = stale_global_step + 1
        if self._next_step is None or self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
        if self._next_step is None:
            self._summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), global_step)
        if self._request_summary:
            self._timer.update_last_triggered_step(global_step)
            if 'summary' in run_values.results:
                for summary in run_values.results['summary']:
                    self._summary_writer.add_summary(summary, global_step)
                    self._summary_writer.flush()
        self._next_step = global_step

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()

    def _get_summary_op(self):
        summary_op = None
        if self._summary_op is not None:
            summary_op = self._summary_op
        elif self._scaffold.summary_op is not None:
            summary_op = self._scaffold.summary_op

        if summary_op is None:
            return None

        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op


class DebugDataHook(tf.train.SessionRunHook):
    def __init__(self, data, vocab, steps_per_debug):
        self._data = data
        self._word2idx = vocab['word2idx']
        self._idx2word = vocab['idx2word']
        self._steps_per_debug = steps_per_debug

    def begin(self):
        self._global_step = tf.train.get_global_step()

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step)

    def after_run(self, run_context, run_values):
        step_value = int(run_values.results)
        if step_value % self._steps_per_debug == 0 and step_value != 0:
            batch_data = run_context.session.run(self._data)
            data_ids = {k: v[0] for k, v in batch_data.items()}
            data_words = {k: [self._idx2word[x] for x in v if x != 0] for k, v in data_ids.items()}
            output = ['%s: %s' % (k, ' '.join(v).encode('utf8')) for k, v in data_words.items()]
            logger.info('\n%s' % ('\n'.join(output)))

