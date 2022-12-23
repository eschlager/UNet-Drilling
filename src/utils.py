# -*- coding: utf-8 -*-
"""
Created on 01.09.2022
@author: eschlager

Functions for computing predictions, metrics and creating plots of the results
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss  # https://pypi.org/project/focal-loss/


def predict_cat(model, x, batch_size, thresh=.5):
    """
    makes prediction as probabilities in [0,1] and categorical {0,1} (OHE if nr classes > 1)
    """
    pred = model.predict(x, batch_size)
    classes = pred.shape[-1]
    if classes == 1:
        pred_cat = (pred > thresh).astype(float)
    elif classes > 1:
        pred_cat = tf.keras.utils.to_categorical(np.argmax(pred, axis=-1), classes)
    return pred, pred_cat


## Loss Functions
def loss_functions(loss_spec, weights=[1., 1., 1.]):
    if loss_spec == 'cross_entropy':
        def c_loss(truth, pred):
            if truth.shape[-1] == 1:
                my_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(truth, pred)
                return my_loss
            elif truth.shape[-1] == 3:
                my_loss = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(truth, pred)
                return my_loss

    elif loss_spec == 'focal_cross_entropy':
        def c_loss(truth, pred):
            if truth.shape[-1] == 1:
                my_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, reduction=tf.keras.losses.Reduction.NONE)(
                    truth, pred)
                return my_loss
            elif truth.shape[-1] == 3:
                truth_classes = tf.keras.backend.argmax(truth, axis=-1)
                my_loss = SparseCategoricalFocalLoss(gamma=2.0, reduction=tf.keras.losses.Reduction.NONE)(
                    truth_classes, pred)
                return my_loss

    elif loss_spec == 'iou_loss':
        def c_loss(truth, pred):
            my_loss = iou_tf(tf.cast(truth, tf.float32), pred)
            # replace nans by 1:
            indices = tf.where(tf.math.is_nan(my_loss))
            my_loss = tf.tensor_scatter_nd_update(my_loss, indices, tf.ones((tf.shape(indices)[0])) * 1.)
            if truth.shape[-1] == 3:
                my_loss = tf.nn.weighted_moments(my_loss, axes=[-1], frequency_weights=weights)[0]
            return 1 - my_loss

    return c_loss


def iou_tf(truth, pred):
    """
    iou tensor for loss computation
    """
    if truth.shape != pred.shape:
        raise Exception(f'truth and pred must have same shape! got {truth.shape} and {pred.shape}')

    tp, fn, fp, tn = confusion_matrix(truth, pred)

    intersection = tp
    union = fp + fn + tp

    iou = tf.math.divide(intersection, union)

    return tf.squeeze(iou)


def confusion_matrix(truth, pred):
    """
    Compute tp, fn, fp and tn
    when using probability predictions for pred these are "soft versions" [see Miao2021]
    """
    tp = tf.reduce_sum(truth * pred, axis=(1, 2))  # true positives
    fn = tf.reduce_sum(truth * (1 - pred), axis=(1, 2))  # false negatives
    fp = tf.reduce_sum((1 - truth) * pred, axis=(1, 2))  # false positives
    tn = tf.reduce_sum((1 - truth) * (1 - pred), axis=(1, 2))  # true negatives
    return tp, fn, fp, tn


## Performance & Evaluation Metrics
def sensitivity(truth, pred, average=None, reduce=True):
    """compute sensitivity (true positive rate)
       for multiclass data:
       - average=None: the scores for each class are returned
       - average='micro': Calculate metrics globally by counting the total true positives, false negatives and false
                          positives
       - average='macro': Calculate metrics for each label, and find their unweighted mean. This does not take label
                          imbalance into account.
    Attention: For samples with masks without any true label the tpr is set to 0
    """
    if truth.shape != pred.shape:
        raise Exception(f'truth and pred must have same shape! got {truth.shape} and {pred.shape}')

    if average == 'micro':
        axis = (1, 2, 3)
    else:
        axis = (1, 2)

    tp = np.sum((truth == 1) & (pred == 1), axis=axis)
    trues = np.sum((truth == 1), axis=axis)
    tpr = np.divide(tp, trues, out=np.empty(tp.shape, dtype=float) * np.nan, where=trues != 0)

    if average == 'macro':
        tpr = np.nanmean(tpr, axis=-1)

    if reduce:
        tpr = np.nanmean(tpr, axis=0)  # mean over all samples
    return tpr.squeeze()


def specificity(truth, pred, average=None, reduce=True):
    """compute sensitivity (true negative rate)
       for multiclass data:
       - average=None: the scores for each class are returned
       - average='micro': Calculate metrics globally by counting the total true positives, false negatives and false
                          positives
       - average='macro': Calculate metrics for each label, and find their unweighted mean. This does not take label
                          imbalance into account.
    Attention: For samples with masks without any true label the tpr is set to 0
    """
    if truth.shape != pred.shape:
        raise Exception(f'truth and pred must have same shape! got {truth.shape} and {pred.shape}')

    if average == 'micro':
        axis = (1, 2, 3)
    else:
        axis = (1, 2)

    tn = np.sum((truth == 0) & (pred == 0), axis=axis)
    trues = np.sum((truth == 0), axis=axis)

    tnr = np.divide(tn, trues, out=np.empty(tn.shape, dtype=float) * np.nan, where=trues != 0)

    if average == 'macro':
        tnr = np.nanmean(tnr, axis=-1)

    if reduce:
        tnr = np.nanmean(tnr, axis=0)  # mean over all samples
    return tnr.squeeze()


def intersection_over_union(truth, pred, average=None, reduce=True):
    """compute mean of intersection over union for a batch of predictions
     soft scores if predictions are probability values
     for multiclass data:
       - average=None: the scores for each class are returned
       - average='micro': Calculate metrics globally by counting the total true positives, false negatives and false
                          positives
       - average='macro': Calculate metrics for each label, and find their unweighted mean. This does not take label
                          imbalance into account.
    """
    if truth.shape != pred.shape:
        raise Exception(f'truth and pred must have same shape! got {truth.shape} and {pred.shape}')

    tp, fn, fp, tn = confusion_matrix(truth, pred)

    intersection = tp
    union = fp + fn + tp

    if average == 'micro':
        intersection = tf.reduce_sum(intersection, axis=-1)
        union = tf.reduce_sum(union, axis=-1)

    iou = tf.math.divide(intersection, union)  # may include nans
    if average == 'macro':
        iou = tf.experimental.numpy.nanmean(iou, axis=-1)

    if reduce:
        iou = tf.experimental.numpy.nanmean(iou, axis=0)

    return iou.numpy().squeeze()


def dice_coef(truth, pred, smooth=1, average=None, reduce=True):
    """compute mean of intersection over union for a batch of predictions
     soft scores if predictions are probability values
     for multiclass data:
       - average=None: the scores for each class are returned
       - average='micro': Calculate metrics globally by counting the total true positives, false negatives and false
                          positives
       - average='macro': Calculate metrics for each label, and find their unweighted mean. This does not take label
                          imbalance into account.
    """
    if truth.shape != pred.shape:
        raise Exception(f'truth and pred must have same shape! got {truth.shape} and {pred.shape}')

    tp, fn, fp, tn = confusion_matrix(truth, pred)

    intersection = tp
    union = 2. * tp + fp + fn
    if average == 'micro':
        dice = (2. * np.sum(intersection, axis=-1) + smooth) / (np.sum(union, axis=-1) + smooth)
    else:
        dice = (2. * intersection + smooth) / (union + smooth)

    if average == 'macro':
        dice = np.mean(dice, axis=-1)

    if reduce:
        dice = np.mean(dice, axis=0)  # mean over all samples
    return dice.squeeze()


## Plotting functions

def plot_image_vs_mask_vs_pred(image, mask_real, mask_pred, wear_mode, arrange='h'):
    image_plus_real_mask = add_mask(image.copy(), mask_real, wear_mode)
    image_plus_pred_mask = add_mask(image.copy(), mask_pred, wear_mode)

    if arrange == 'h':
        fig, axs = plt.subplots(ncols=3, figsize=(12, 4.2))
    elif arrange == 'v':
        fig, axs = plt.subplots(nrows=3, figsize=(12, 12))

    axs[0].set_title('image')
    axs[0].imshow(image)
    axs[0].axis('off')

    axs[1].set_title('real mask')
    axs[1].imshow(image_plus_real_mask)
    axs[1].axis('off')

    iou = intersection_over_union(np.expand_dims(mask_real, axis=0), np.expand_dims(mask_pred, axis=0), reduce=False)
    axs[2].set_title('predicted mask')
    axs[2].imshow(image_plus_pred_mask)
    if wear_mode == 0:
        axs[2].text(0.03, 0.03, f'IoU = {round(iou.item(), 2)}', fontsize=20, color=(1, 1, 0),
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='k')],
                    transform=axs[2].transAxes)
    elif wear_mode == 1:
        axs[2].text(0.03, 0.03, f'IoU = {round(iou.item(), 2)}', fontsize=20, color=(0, .56, 1),
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='k')],
                    transform=axs[2].transAxes)
    elif wear_mode == 2:
        axs[2].text(0.03, 0.03, f'IoU = {round(iou.item(), 2)}', fontsize=20, color=(1, 0, 1),
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='k')],
                    transform=axs[2].transAxes)
    elif wear_mode == 3:
        axs[2].text(0.03, 0.15, f'IoU (a)= {round(iou[1], 2)}', fontsize=20, color=(1, 1, 0),
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='k')],
                    transform=axs[2].transAxes)
        axs[2].text(0.03, 0.03, f'IoU (b) = {round(iou[2], 2)}', fontsize=20, color=(0, .56, 1),
                    path_effects=[path_effects.withStroke(linewidth=3, foreground='k')],
                    transform=axs[2].transAxes)
    axs[2].axis('off')
    plt.tight_layout()
    return fig


def add_mask_batch(img_batch, mask_batch, wear_mode):
    pred_batch = np.zeros(img_batch.shape)
    for nr, (img, mask) in enumerate(zip(img_batch, mask_batch)):
        pred_batch[nr] = add_mask(img, mask, wear_mode)
    return pred_batch


def add_mask(img, mask, wear_mode):
    img_masked = img.copy()
    if wear_mode == 0:
        img_masked[mask[:, :, 0] == 1] = [1, 1, 0]
    elif wear_mode == 1:
        img_masked[mask[:, :, 0] == 1] = [0, .56, 1]
    elif wear_mode == 2:
        img_masked[mask[:, :, 0] == 1] = [1, 0, 1]
    elif wear_mode == 3:
        img_masked[mask[:, :, 1] == 1] = [1, 1, 0]
        img_masked[mask[:, :, 2] == 1] = [0, .56, 1]
    return img_masked


def plot_history(history):
    fig, ax = plt.subplots()
    ax.set_xlabel('epochs')
    ax.plot(history['loss'], color='black')
    ax.set_ylabel('train loss')
    ax2 = ax.twinx()
    ax2.plot(history['val_loss'], color='grey')
    ax2.set_ylabel('val loss')
    ax2.yaxis.label.set_color("grey")
    plt.tight_layout()
    return fig
