# -*- coding: utf-8 -*-

"""
Created on 18.08.2022
@author: eschlager

Custom callback functions for U-Net

"""

import tensorflow as tf
import logging
import keras.backend as K


def reset_weights(model, lr):
    """ reinitialize a keras model
    from https://stackoverflow.com/questions/40496069/reset-weights-in-keras-layer
    """
    model.optimizer.lr = lr
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer

            old_weights, old_biases = model.layers[ix].get_weights()

            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=old_biases.shape)])


class MyLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info('Epoch %03d - loss: %.4f - val_loss %.4f - lr %.6e' % (
            epoch, logs['loss'], logs['val_loss'], K.eval(self.model.optimizer.lr)))
