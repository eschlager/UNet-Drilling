# -*- coding: utf-8 -*-
"""
Created on 18.10.2022
@author: eschlager


Evaluation of a model whole image with overlap-tile strategy 

"""

import argparse
import logging
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pickle
import tiler

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.sep.join([script_dir, '..', 'src']))
from data_loader import DataLoader
import logging_config
import utils

home_dir = os.path.abspath(os.path.join(script_dir, '..'))


class ModelPredictor:
    def __init__(self, path_to_model, batch_size):
        self.path_to_model = path_to_model

        self.model = self.load_unet_model()
        self.tile_shape = self.model.input_shape[1]
        self.wear_mode = self.get_wear_mode()
        self.batch_size = batch_size
        if self.wear_mode == 3:
            self.nr_classes = 3
        else:
            self.nr_classes = 1

    def load_unet_model(self):
        logging.info(f"Load model {os.path.abspath(self.path_to_model)}")
        model = tf.keras.models.load_model(self.path_to_model, compile=False)
        return model

    def get_wear_mode(self):
        model_args = pickle.load(open(os.path.join(self.path_to_model, "..", "arguments.pkl"), 'rb'))
        wear_mode = model_args["wear_mode"]
        logging.info(f"Model wear mode: {wear_mode}")
        return wear_mode

    def predict_image(self, img):
        img_tiler = tiler.Tiler(
            data_shape=img.shape,
            tile_shape=(self.tile_shape, self.tile_shape, img.shape[-1]),
            overlap=(184, 184, 0),
            channel_dimension=2,
        )

        mask_tiler = tiler.Tiler(
            data_shape=img.shape,
            tile_shape=(self.tile_shape, self.tile_shape, self.nr_classes),
            overlap=(184, 184, 0),
            channel_dimension=2,
        )

        logging.info("  Perform image padding...")
        new_shape, padding = img_tiler.calculate_padding()
        img_tiler.recalculate(data_shape=new_shape)
        mask_tiler.recalculate(data_shape=new_shape)
        padded_img = np.pad(img, padding, mode="reflect")

        mask_merger = tiler.Merger(tiler=mask_tiler, window="overlap-tile")

        logging.info(f"  Perform prediction for all {len(img_tiler)} tiles in batches of size {self.batch_size}...")
        for batch_id, batch in img_tiler(padded_img, progress_bar=True, batch_size=self.batch_size):
            _, pred_cat = utils.predict_cat(self.model, batch, self.batch_size)
            mask_merger.add_batch(batch_id, self.batch_size, pred_cat)

        mask_pred = mask_merger.merge(extra_padding=padding, dtype=img.dtype)

        return mask_pred


def scale_img(img, width_new, height_new):
    fig = plt.figure(figsize=(width_new, height_new))
    ax = plt.Axes(fig, [0., 0., 1., 1.])

    fig.add_axes(ax)
    ax.axis('off')
    ax.imshow(img)
    return ax


def main(gpu_id, data_path, model_path, batch_size):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # directory of models to evaluate
    model_dir = os.path.join(home_dir, model_path)

    out_dir = os.path.join(home_dir, "evaluation",
                           'model_' + os.path.basename(os.path.dirname(model_dir)),
                           'images_' + os.path.basename(os.path.normpath(data_path)) + '_' +
                           os.path.basename(os.path.normpath(model_dir)))
    os.makedirs(out_dir, exist_ok=True)

    logging_config.define_root_logger(os.path.join(out_dir, f"log_predict_images.txt"))
    logging.info(f"Output directory {os.path.abspath(out_dir)}")

    predictor = ModelPredictor(model_dir, batch_size)

    logging.info("Starting Image Wear Predictions...")
    data_dir = os.path.join(home_dir, data_path)
    files = os.listdir(data_dir)
    files_img = np.array([e for e in files if not '_masked' in e])
    n_samples = len(files_img)
    logging.info(f'Found {n_samples} images to load.')

    loader = DataLoader(data_dir, None, predictor.wear_mode)

    figwidth = 5
    _dpi = 300

    for file in os.listdir(data_dir):
        img_name, ext = os.path.splitext(file)
        if not img_name.endswith('_masked'):
            logging.info(f'Make predictions for image {file}')

            # start_time_fit = time.time()
            img = loader.load_image(file)
            height, width, _ = img.shape
            scaler = width / figwidth
            height_new = height / scaler
            width_new = width / scaler

            # scale and save image
            ax_img = scale_img(img, width_new, height_new)
            plt.savefig(os.path.sep.join([os.path.abspath(out_dir), f"{img_name}.jpg"]), dpi=_dpi, pad_inches=0,
                        bbox_inches='tight')
            plt.close()

            # scale and save image with true mask
            mask = loader.load_mask(img_name + '_masked' + ext)
            image_plus_real_mask = utils.add_mask(img.copy(), mask, predictor.wear_mode)
            ax_mask = scale_img(image_plus_real_mask, width_new, height_new)
            plt.savefig(os.path.sep.join([os.path.abspath(out_dir), f"{img_name}_true.jpg"]), dpi=_dpi, pad_inches=0,
                        bbox_inches='tight')
            plt.close()

            # create prediction, scale image, and add IoU to image
            pred_cat = predictor.predict_image(img)
            image_plus_pred_mask = utils.add_mask(img.copy(), pred_cat, predictor.wear_mode)
            ax_pred = scale_img(image_plus_pred_mask, width_new, height_new)

            if predictor.wear_mode == 3:
                mask = 1 - mask[:, :, :1]
                pred_cat = 1 - pred_cat[:, :, :1]
            iou = utils.intersection_over_union(np.expand_dims(mask, axis=0), np.expand_dims(pred_cat, axis=0),
                                                reduce=False)
            ax_pred.text(0.05, 0.05, f'IoU = {round(iou.item(), 2)}', fontsize=20, color=(1, 0, 1),
                         path_effects=[path_effects.withStroke(linewidth=2, foreground='k')],
                         transform=ax_pred.transAxes)
            plt.savefig(os.path.sep.join([os.path.abspath(out_dir), f"{img_name}_pred.jpg"]), dpi=_dpi, pad_inches=0,
                        bbox_inches='tight')
            plt.close()


parser = argparse.ArgumentParser(description="Image Wear Prediction")
parser.add_argument('--gpu_id', default="3,4", type=str)
parser.add_argument("-i", "--data_path", default="data/raw/dev",
                    type=str, help="folder of images to be evaluated")
parser.add_argument("-m", "--model_path",
                    default="models/finalbn_alittleaug_iou_3_00/trained_model_fold00.tf",
                    type=str, help="directory to model")
parser.add_argument("-b", "--batch_size", default=32,
                    type=int, help="batch size used for predicting.")

args = parser.parse_args()

if __name__ == "__main__":
    main(args.gpu_id, args.data_path, args.model_path, args.batch_size)
