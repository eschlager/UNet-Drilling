# -*- coding: utf-8 -*-
"""
Created on 18.08.2022
@author: eschlager

Training of U-Net using cross validation

"""

import argparse
import logging
import math
import os
import pickle
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

import matplotlib

matplotlib.use('Agg')

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.sep.join([script_dir, 'src']))
sys.path.append(os.path.sep.join([script_dir, 'unet']))
sys.path.append(os.path.sep.join([script_dir, 'evaluate']))

import logging_config
import utils
from data_loader import DataLoader
import unet
from utils_unet import reset_weights, MyLogger
import evaluate_cv

from tensorflow.python.data.util import options as options_lib
from tensorflow.data.experimental import DistributeOptions, AutoShardPolicy

DistributeOptions.auto_shard_policy = options_lib.create_option(
    name="auto_shard_policy",
    ty=AutoShardPolicy,
    docstring="The type of sharding to use. See "
              "`tf.data.experimental.AutoShardPolicy` for additional information.",
    default_factory=lambda: AutoShardPolicy.DATA,
)


home_dir = script_dir

def build_network(input_shape, num_classes):
    model = unet.U_Net(input_shape, num_classes)
    return model


def load_data(_dir, _size, _wear_mode):
    data = DataLoader(_dir, _size, _wear_mode)
    images, masks = data.load_imgs_and_masks()
    return images, masks


def data_generator(images, masks, batch_size, epochs):
    dataset = tf.data.Dataset.from_tensor_slices((images, masks)).shuffle(50, reshuffle_each_iteration=True).repeat(
        epochs).batch(batch_size)
    return dataset


def main(args):
    start_time = time.time()

    out_dir = os.sep.join([home_dir, args.model_path])
    os.makedirs(out_dir, exist_ok=True)
    logging_config.define_root_logger(os.path.join(out_dir, f'log.txt'))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    strategy = tf.distribute.MirroredStrategy()
    logging.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    logging.info(f'Use loss function {args.loss}.')
    weights = None
    if args.wear_mode == 3 and args.loss == 'iou_loss':
        weights = [0.2, 1.4, 1.4]
        logging.info(f'Used weights of averaging {weights}.')

    # Load data
    in_dir = os.sep.join([home_dir, args.data_path])
    size = int(in_dir.split('_')[-1])
    images, masks = load_data(in_dir, size, args.wear_mode)

    logging.info(f'Loaded image data: {images.shape}')
    logging.info(f'Loaded mask data: {masks.shape}')
    num_classes = masks.shape[-1]

    with strategy.scope():
        model = build_network((size, size, 3), num_classes)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                      loss=utils.loss_functions(args.loss, weights=weights)
                      )
    model.summary(print_fn=logging.info)

    if args.num_folds > 1:
        # train with Kfold cross-validation
        kfold = KFold(n_splits=args.num_folds, shuffle=True)
        folds = kfold.split(images, masks)
    else:
        # if no cross-validation, use 10% of data for validation
        indices = np.arange(len(images))
        random.shuffle(indices)
        train_idc, val_idc = np.split(indices, [math.floor(len(images) * 0.9)], axis=0)
        folds = [(train_idc, val_idc)]

    # dict for monitoring loss, IoU and dice of train and validation set
    kfold_res = {'train_loss': [], 'val_loss': [], 'train_IOU': [], 'val_IOU': [], 'train_dice': [], 'val_dice': []}
    if num_classes > 1:
        kfold_res_class = {'train_IOU_class': [], 'val_IOU_class': [], 'train_dice_class': [], 'val_dice_class': []}
    hist_res = []

    logging.info(f'Perform training with {args.num_folds}-fold cross validation using following arguments: ')
    for key, val in vars(args).items():
        logging.info(f'    {key}: {val}')
    args_file_name = f'{out_dir}/arguments.pkl'
    logging.info(f'Save used hyper parameters in {args_file_name}.')
    pickle.dump(vars(args), open(args_file_name, 'wb'))

    for ifold, (train, val) in enumerate(folds):
        logging.info(f"Reset weights and learning rate for fold {ifold:02}.")
        # reset_weights(model, args.learning_rate)
        images_train = images[train]
        masks_train = masks[train]
        images_val = images[val]
        masks_val = masks[val]

        logging.info(f"Shape of train data: {images_train.shape}")
        logging.info(f"Shape of val data: {images_val.shape}")

        train_data = data_generator(images_train, masks_train, args.batch_size, args.epochs).with_options(options)

        logging.info(f"Start training of fold {ifold:02}.")

        my_callbacks = [MyLogger(),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=5, verbose=1,
                                                             mode="min",
                                                             min_delta=0.005, min_lr=1e-6)]

        if args.early_stopping:
            my_callbacks = my_callbacks + [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                                 patience=args.early_stop_patience, start_from_epoch=60)]

        if args.restart:
            trials = 5
        else:
            trials = 1
        tried = 0

        while tried < trials:
            tried += 1
            logging.info(f"Start training for {tried}. time:")
            reset_weights(model, args.learning_rate)

            start_time_fit = time.time()
            hist = model.fit(train_data,
                             epochs=args.epochs,
                             steps_per_epoch=images_train.shape[0] // args.batch_size,
                             validation_data=(images_val, masks_val),
                             validation_batch_size=16,
                             callbacks=my_callbacks)
            dur_fit = time.time() - start_time_fit

            if hist.history['val_loss'][-1] < args.restart_thresh:
                tried = trials
                logging.info(f"Training was successful!")
            elif tried == trials:
                logging.info(f"Started training {trials} times; cannot get below threshold of {args.restart_thresh}.")

        logging.info(f"Time needed to fit fold {ifold:02}: {dur_fit}.")

        logging.info(f"Save model of fold {ifold:02} in {out_dir}.")
        model.save(os.path.sep.join([out_dir, f'trained_model_fold{ifold:02}.tf']))
        hist_res.append(hist)
        pickle.dump([e.history for e in hist_res], open(os.path.sep.join([out_dir, 'training_hists.pkl']), 'wb'))

        # evaluate and save scores
        logging.info(f"Evaluate model of fold {ifold:02}.")
        pred_train, pred_train_cat = utils.predict_cat(model, images_train, batch_size=1)
        train_loss = tf.math.reduce_mean(utils.loss_functions(args.loss)(masks_train, pred_train)).numpy()
        logging.info(f"Training loss: {train_loss.squeeze():.4f}.")
        kfold_res['train_loss'].append(train_loss)
        train_ce = tf.math.reduce_mean(utils.loss_functions('cross_entropy')(masks_train, pred_train)).numpy()
        logging.info(f"Training cross entropy: {train_ce.squeeze():.4f}.")
        train_iou = np.around(utils.intersection_over_union(masks_train, pred_train_cat), 4)
        logging.info(f"Training IOU: {train_iou.squeeze()}.")
        kfold_res['train_IOU'].append(np.mean(train_iou))
        train_dice = np.around(utils.dice_coef(masks_train, pred_train_cat), 4)
        logging.info(f"Training dice coefficient: {train_dice.squeeze()}.")
        kfold_res['train_dice'].append(np.mean(train_dice))
        if num_classes > 1:
            kfold_res_class['train_IOU_class'].append(train_iou)
            kfold_res_class['train_dice_class'].append(train_dice)

        pred_val, pred_val_cat = utils.predict_cat(model, images_val, batch_size=1)
        val_loss = tf.math.reduce_mean(utils.loss_functions(args.loss)(masks_val, pred_val)).numpy()
        logging.info(f"Validation loss: {val_loss.squeeze():.4f}.")
        kfold_res['val_loss'].append(val_loss)
        val_ce = tf.math.reduce_mean(utils.loss_functions('cross_entropy')(masks_val, pred_val)).numpy()
        logging.info(f"Validation cross entropy: {val_ce.squeeze():.4f}.")
        val_iou = np.around(utils.intersection_over_union(masks_val, pred_val_cat), 4)
        logging.info(f"Validation IOU: {val_iou.squeeze()}.")
        kfold_res['val_IOU'].append(np.mean(val_iou))
        val_dice = np.around(utils.dice_coef(masks_val, pred_val_cat), 4)
        logging.info(f"Validation dice coefficient: {val_dice.squeeze()}.")
        kfold_res['val_dice'].append(np.mean(val_dice))
        if num_classes > 1:
            kfold_res_class['val_IOU_class'].append(val_iou)
            kfold_res_class['val_dice_class'].append(val_dice)

        fig = utils.plot_history(hist.history)
        plt.savefig(os.path.sep.join([out_dir, f'history_fold{ifold:02}.jpg']), bbox_inches='tight')
        plt.close()

        if args.do_plot:
            # plot original image, image with real mask, and image with predicted mask
            for i in range(len(val)):
                fig = utils.plot_image_vs_mask_vs_pred(images_val[i], masks_val[i], pred_val_cat[i], args.wear_mode)
                plt.savefig(os.path.sep.join([out_dir, f'predicted_fold{ifold:02}_{i:02}.png']), bbox_inches='tight')
                plt.close()

        logging.info(f'Cross validation finished - save results to {out_dir}')
        with pd.ExcelWriter(os.path.sep.join([out_dir, 'model_scores.xlsx'])) as writer:
            df_res = pd.DataFrame(kfold_res)
            df_res.to_excel(writer, sheet_name='results', float_format="%.4f")
            if num_classes > 1:
                df_res_class = pd.DataFrame()
                for key in kfold_res_class.keys():
                    df_key = pd.DataFrame(kfold_res_class[key])
                    df_key.columns = [key + '_' + str(nr) for nr in df_key.columns]
                    df_res_class = df_res_class.join(df_key, how='outer')
                df_res_class.to_excel(writer, sheet_name='results_class', float_format="%.4f")

    dur_total = time.time() - start_time
    logging.info(f"Total time needed: {dur_total}.")
    tf.keras.backend.clear_session()

    # apply evaluation on images in eval_data_folder
    eval_data_folder = os.path.join(os.path.dirname(args.data_path), f'dev_centertiles_{size}')
    evaluate_cv.main(args.gpu_id, args.model_path, eval_data_folder)

    return locals()


def set_parser():
    """
    Define argument parser with default arguments
    """
    parser = argparse.ArgumentParser(description='train_unet')
    parser.add_argument('--gpu_id', default="1,2", type=str)
    parser.add_argument('--data_path', default='data/interim/train_alittleaugmented_512', type=str,
                        help='Input path of train data.')
    parser.add_argument('--model_path', default='models/unetbn_alittleaug_iou_3_00', type=str,
                        help='Path where trained models are stored.')
    parser.add_argument('--wear_mode', default=3, type=int, help='Define which wear types should be trained: ´\n'
                                                                 '0 - build-up-edge\n'
                                                                 '1 - abrasive wear\n'
                                                                 '2 - both as one label\n'
                                                                 '3 - train both as separate labels.')
    parser.add_argument('--loss', default='iou_loss', type=str,
                        help="cross_entropy, focal_cross_entropy, or iou_loss.")
    parser.add_argument('--num_folds', default=1, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--restart', default=True, type=bool, help="True or False")
    parser.add_argument('--restart_thresh', default=.4, type=float)
    parser.add_argument('--early_stopping', default=True, type=bool, help="True or False")
    parser.add_argument('--early_stop_patience', default=20, type=int)
    parser.add_argument('--do_plot', default=False, type=bool, help="True or False")

    return parser


if __name__ == "__main__":
    parser = set_parser()
    args = parser.parse_args()
    locals().update(main(args))
