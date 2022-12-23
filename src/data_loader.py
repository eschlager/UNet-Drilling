# -*- coding: utf-8 -*-
"""
Created on 19.08.2022
@author: eschlager
"""

import glob
import logging
import os

import image_to_numpy
import numpy as np
import tqdm
import sys


class DataLoader(object):
    def __init__(self, directory, size, wear_mode):
        self.dir = directory
        self.size = size
        self.wear_mode = wear_mode

        self.img_ext = 'png'
        self.mask_ext = 'png'

        if not os.path.isdir(self.dir):
            logging.error(f'Directory {self.dir} does not exist!')
            sys.exit(1)
        else:
            logging.info(f'Load data from directory {self.dir}.')

        self.init_wear()

    def init_wear(self):
        if self.wear_mode == 0:
            logging.info('Operate in wear_mode 0: segment build-up-edge only')
            self.nr_classes = 1
        elif self.wear_mode == 1:
            logging.info('Operate in wear_mode 1: segment abrasive wear only')
            self.nr_classes = 1
        elif self.wear_mode == 2:
            logging.info('Operate in wear_mode 2: segment build-up-edge and abrasive wear together as one mask')
            self.nr_classes = 1
        elif self.wear_mode == 3:
            logging.info('Operate in wear_mode 3: segment build-up-edge and abrasive wear as multiclass problem.')
            self.nr_classes = 3  # +1 class due to background

    def load_imgs_and_masks(self):
        """
        Load all images with associated masks from directory self.dir.
        Images need to be RGB and are scaled to valued in [0,1]
        Masks are converted to binary (OHE)
        """
        fnames_all = sorted(glob.glob(f'{self.dir}/*.{self.img_ext}'))
        fnames_orig = np.array([e for e in fnames_all if not '_masked' in e])
        n_samples = len(fnames_orig)
        logging.info(f'Found {n_samples} images to load.')

        # initialize array that will hold all data
        images = np.zeros([n_samples] + [self.size, self.size] + [3], dtype='float32')
        masks = np.zeros([n_samples] + [self.size, self.size] + [self.nr_classes], dtype='int')

        for i in tqdm.trange(n_samples):
            images[i], masks[i] = self.load_and_prep_sample(fnames_orig[i])
        return images, masks

    def load_image(self, img_name):
        image = image_to_numpy.load_image_file(os.path.join(self.dir, img_name))
        return image / 255.  # Scale RGB values to [0,1]

    def load_mask(self, mask_name):
        mask = image_to_numpy.load_image_file(os.path.join(self.dir, mask_name))
        mask = self.convert_mask(mask)
        return mask

    def load_and_prep_sample(self, fname):
        """
        Load image and associated mask named fname from directory self.dir
        Mask have to have same name with suffix '_masked'
        """
        image = self.load_image(fname)
        name, ext = os.path.splitext(fname)
        fname_mask = name + '_masked.' + self.mask_ext
        mask = image_to_numpy.load_image_file(fname_mask)

        if self.size:
            if not image.shape[0] == image.shape[1] == self.size:
                raise Exception(f'Image must be square with size {self.size}, but is of shape {image.shape}.')
        if not (image.shape[0] == mask.shape[0] and image.shape[1] == mask.shape[1]):
            raise Exception(
                f'Image and mask dimension do not fit together: image of shape {image.shape}, mask of shape {mask.shape}.')

        mask = self.convert_mask(mask)

        return image, mask

    def convert_mask(self, mask):
        """
        converts RGB-coloured masks into binary (OHE) masks; depends on the labelling used.
        here original masks are labelled the following:
        - wear type 0 is yellow=(248, 249, 0) --> check mask[:, :, 0] == 248
        - wear type 1 is blue=(0, 143, 255) --> check mask[:, :, 2] == 255
        - wear type 2 is yellow or blue --> check mask[:, :, 1] > 0
        """
        mask_decoded = np.zeros([mask.shape[0], mask.shape[1]] + [self.nr_classes], dtype='int')

        if self.wear_mode == 0:
            mask_decoded[:, :, 0] = mask[:, :, 0] == 248
        elif self.wear_mode == 1:
            mask_decoded[:, :, 0] = mask[:, :, 2] == 255
        elif self.wear_mode == 2:
            mask_decoded[:, :, 0] = mask[:, :, 1] > 0
        elif self.wear_mode == 3:
            mask_decoded[:, :, 1] = mask[:, :, 0] == 248
            mask_decoded[:, :, 2] = mask[:, :, 2] == 255
            mask_decoded[:, :, 0] = (mask_decoded[:, :, 1] + mask_decoded[:, :, 2]) == 0  # background

        return mask_decoded
