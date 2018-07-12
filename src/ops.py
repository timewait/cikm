import tensorflow as tf

def matrix_hingle_loss(scores, margin=0.1):
    diag_scores = tf.diag_part(scores)
    dim = tf.shape(scores)[0]
    diag_scores_matrix = tf.transpose(tf.reshape(tf.tile(diag_scores, [dim]), [dim, dim]))
    loss = tf.reduce_mean(tf.nn.relu(margin - (diag_scores_matrix - scores)))
    return loss


def cosine_similarity(a, b, mode='matrix'):
    # make sure no zero vector in a and b, otherwise nan error
    a_norm = a / tf.norm(a, axis=1, keep_dims=True)
    b_norm = b / tf.norm(b, axis=1, keep_dims=True)
    if mode == 'vector':
        cos_sim = tf.reduce_sum(a_norm * b_norm, axis=1)
    elif mode == 'matrix':
        cos_sim = tf.matmul(a_norm, b_norm, transpose_b=True)
    else:
        raise ValueError('mode must be vector or matrix.')
    return cos_sim


def get_optimizer_fn(name):
    name = name.lower()
    if name == 'adam':
        opt_fn = tf.train.AdamOptimizer
    elif name == 'sgd':
        opt_fn = tf.train.GradientDescentOptimizer
    elif name == 'rmsprop':
        opt_fn = tf.train.RMSPropOptimizer
    elif name == 'adadelta':
        opt_fn = tf.train.AdadeltaOptimizer
    elif name == 'adagrad':
        opt_fn = tf.train.AdagradOptimizer
    else:
        raise ValueError

    return opt_fn


def get_activation_fn(name):
    name = name.lower()
    if name == 'tanh':
        act_fn = tf.nn.tanh
    elif name == 'softsign':
        act_fn = tf.nn.softsign
    elif name == 'relu':
        act_fn = tf.nn.relu
    elif name == 'sigmoid':
        act_fn = tf.nn.sigmoid
    elif name == 'none':
        act_fn = tf.identity
    else:
        raise ValueError

    return act_fn
