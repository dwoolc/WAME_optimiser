from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer

from six.moves import zip

if K.backend() == 'tensorflow':
    import tensorflow as tf

class WAME(Optimizer):
    ''' WAME optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor.
    This code is used with reference to source at
    https://github.com/nitbix/keras-oldfork/blob/master/keras/optimizers.py
    '''


    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-11, decay=0., eta_plus = 1.2, eta_minus = 0.1,
                 eta_min=1e-2, eta_max=1e2, beta_a = 0.9,
                 **kwargs):
        learning_rate = kwargs.pop('lr', lr)
        super(WAME, self).__init__(**kwargs)
        self.__dict__.update(locals())
        self.iterations = K.variable(0)
        self.learning_rate = K.variable(learning_rate)
        self.beta_1 = K.variable(beta_1)
        self.beta_2 = K.variable(beta_2)
        self.beta_a = K.variable(beta_a)
        self.decay = K.variable(decay)
        self.eta_plus = K.variable(eta_plus)
        self.eta_minus = K.variable(eta_minus)
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.inital_decay = decay


    @interfaces.legacy_get_updates_support
    @K.symbolic
    def get_updates(self, params, loss):
        grads = self.get_gradients(loss, params)
        self.updates = []
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.learning_rate # code amended from orig
        if self.inital_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1
        lr_t = lr * K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params] # different
        prev_grads = [K.zeros(shape) for shape in shapes]
        prev_params = [K.zeros(shape) for shape in shapes]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]
        accs = [K.ones(shape) for shape in shapes]
        acc_ms = [K.ones(shape) for shape in shapes]
        acc_vs = [K.ones(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + vs

        # key diff is grad changes done within loop.
        # Results in weight or element wise updates rather than overall update
        for param, grad, m, v, a, am, av, prev_grad, prev_param in zip(params, grads, ms, vs, accs,
                acc_ms, acc_vs, prev_grads, prev_params):

            change = prev_grad * grad # grad change
            change_below_zero = K.less(change,0.) # boolean value < 0
            change_above_zero = K.greater(change,0.) # boolean value > 0
            a_t = K.switch(
                change_below_zero,
                a * self.eta_minus,
                K.switch(change_above_zero, a * self.eta_plus, a)
            ) # works as if statement - a_t is a * self.eta_minus if change < 0 else  a * self.eta_plus
            a_clipped = K.clip(a_t, self.eta_min, self.eta_max) # conducts elementwise clipping of acceleration time instance (tn) using eta_min & max
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(grad) # produces v(tn).
            am_t = (self.beta_a * am) + (1. - self.beta_a) * a_clipped
            a_rate = a_clipped / am_t
            p_t = param - lr_t * a_rate * grad / (K.sqrt(v_t) + self.epsilon)

            new_param = p_t

            self.updates.append(K.update(v, v_t))
            self.updates.append(K.update(param, new_param))
            self.updates.append(K.update(prev_grad, param))
            self.updates.append(K.update(a, a_t))
            self.updates.append(K.update(am, am_t))
            self.updates.append(K.update(prev_param, param))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)), # code amended from orig
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'beta_a': float(K.get_value(self.beta_a)),
                  'eta_plus': float(K.get_value(self.eta_plus)),
                  'eta_minus': float(K.get_value(self.eta_minus)),
                  'eta_min': float(self.eta_min),
                  'eta_max': float(self.eta_max),
                  'epsilon': self.epsilon}
        base_config = super(WAME, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))













