# -*- coding: utf-8 -*-
"""
Created on 18.10.2022
@author: eschlager

Create wear predictions of images using overlap-tile strategy

"""

import argparse
import logging
import os
import sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tiler

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.sep.join([script_dir, 'src']))
import logging_config
from data_loader import DataLoader
import utils


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


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    out_dir = os.path.join('predictions',
                           os.path.basename(os.path.dirname(args.model_dir)) + '_' + os.path.basename(
                               os.path.normpath(args.model_dir)),
                           os.path.basename(os.path.normpath(args.image_dir)))
    os.makedirs(out_dir, exist_ok=True)

    logging_config.define_root_logger(os.path.join(out_dir, f"log_predict_images.txt"))
    logging.info(f"Predictions output directory {os.path.abspath(out_dir)}")

    predictor = ModelPredictor(args.model_dir, args.batch_size)

    logging.info("Starting Image Wear Predictions...")

    files = os.listdir(args.image_dir)
    files_img = np.array([e for e in files if not '_masked' in e])
    n_samples = len(files_img)
    logging.info(f'Found {n_samples} images to load.')

    loader = DataLoader(args.image_dir, None, predictor.wear_mode)

    for file in os.listdir(args.image_dir):
        img_name, ext = os.path.splitext(file)
        if not img_name.endswith('_masked'):
            logging.info(f'Make predictions for image {file}')

            start_time_fit = time.time()
            img = loader.load_image(file)
            pred_cat = predictor.predict_image(img)

            if args.eval_mode == 0:
                pred_img = utils.add_mask(np.zeros(img.shape), pred_cat, predictor.wear_mode)
                mask_dir = os.path.join(os.path.abspath(out_dir), f"{img_name}_pred.png")
                logging.info(f"  Save predicted mask as {os.path.abspath(mask_dir)}.")
                plt.imsave(mask_dir, pred_img)
                plt.close()

            if args.eval_mode == 1:
                final_img = utils.add_mask(img, pred_cat, predictor.wear_mode)
                img_dir = os.path.join(os.path.abspath(out_dir), f"{img_name}_imgpred.png")
                logging.info(f"  Save image and predicted mask combined as {img_dir}.")
                plt.imsave(img_dir, final_img)
                plt.close()

            dur_fit = time.time() - start_time_fit
            logging.info(f"Time to load image, predict, and save: {dur_fit}.")


parser = argparse.ArgumentParser(description="Image Wear Prediction")
parser.add_argument('--gpu_id', default="1,2", type=str)
parser.add_argument("-i", "--image_dir", default="data/raw/dev",
                    type=str, help="directory of images")
parser.add_argument("-m", "--model_dir",
                    default="models/finalbn_alittleaug_iou_3_00/trained_model_fold00.tf",
                    type=str, help="directory of model")
parser.add_argument("-b", "--batch_size", default=16,
                    type=int, help="batch size used for predicting.")
parser.add_argument("-e", "--eval_mode", default=0,
                    type=int, help="0: save predicted mask only\n"
                                   "1: save image with predicted wear area only")

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
