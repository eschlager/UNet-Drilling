# -*- coding: utf-8 -*-
"""
Created on 17.08.2022
@author: eschlager

Create image tiles of specified size from original image and the respective mask without starting from left to right.
By default, the tiles are centered vertically around the masked part, without any overlaps of different tiles.
Optionally overlaps, rotation, shifts, changes in contrast, etc. can be made.
No tiles without any labeled area are saved.

"""

import logging
import os
import random
import sys

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.sep.join([script_dir, '..', 'src']))
import logging_config


def tile(filename, dir_in, dir_out, d, overlap=0, rot=None, contrast=None, brightness=None, blur=None, shift=None):
    """
    :param filename: filename of original image; mask must have same name with additional suffix '_masked'
    :param dir_in: directory of images
    :param dir_out: output directory where tiles should be saved
    :param d: edge size of tiles
    :param overlap: overlap between the single tiles as rate between [0,1]
    :param rot: interval from which rotation angle is sampled
    :param contrast: interval from which contrast factor is sampled
    :param brightness: interval from which factor for change in brightness is sampled
    :param blur: interval from which radius for Gaussian blur is sampled
    :param shift: interval from which percentage of d is sampled for vertical and horizontal shift
    """

    logging.info(f'\n\nProcess image {filename}')
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    mask = Image.open(os.path.join(dir_in, f'{name}_masked.png'))  # mask are all delivered as png
    logging.info(f'Process image mask {name}_masked.png')
    w, h = img.size

    if not (overlap >= 0 and overlap <= 1):
        raise Exception(f'Parameter \'overlap\' must be between 0 and 1.')
    else:
        stride = int(d * (1 - overlap))
        logging.info(f'Process images to tiles of size {d} with stride length {stride}.')

    grid_w = range(0, w - w % d, stride)
    logging.info(f'Process {len(grid_w)} tiles:')
    for idx, step in enumerate(grid_w):
        logging.info(f'  Process tile nr {idx}...')
        box = (step, 0, step + d, h)
        positions = np.nonzero(mask.crop(box))

        if not positions[0].any():
            logging.info('  Tile does not contain any labelled part. Skip this tile.')
        else:
            # height at which the wear area is located
            mid_height = positions[0].min() + int((positions[0].max() - positions[0].min()) / 2)

            mask_processed = mask.copy()
            img_processed = img.copy()

            if rot:  # rotate each tile around its center point
                rot_angle = random.randint(rot[0], rot[1])
                logging.info(f'    Perform rotation with angle {rot_angle}.')
                mask_processed = mask_processed.rotate(rot_angle, center=(step + np.floor(d / 2), mid_height))
                img_processed = img_processed.rotate(rot_angle, center=(step + np.floor(d / 2), mid_height))
            if contrast:
                scale_contrast = round(random.uniform(contrast[0], contrast[1]), 2)
                logging.info(f'    Scale contrast with factor {scale_contrast}.')
                img_processed = ImageEnhance.Contrast(img_processed).enhance(scale_contrast)
            if brightness:
                scale_brightness = round(random.uniform(brightness[0], brightness[1]), 2)
                logging.info(f'    Scale brightness with factor {scale_brightness}.')
                img_processed = ImageEnhance.Brightness(img_processed).enhance(scale_brightness)
            if blur:
                sigma = round(random.uniform(blur[0], blur[1]), 2)
                logging.info(f'    Blur image with radius {sigma}.')
                img_processed = img_processed.filter(ImageFilter.GaussianBlur(radius=sigma))
            if shift:
                horizontal_shift = round(random.uniform(shift[0], shift[1]), 2)
                vertical_shift = round(random.uniform(shift[0], shift[1]), 2)
                logging.info(
                    f'    Shift image horizontally by {horizontal_shift}*{d} and vertically {vertical_shift}*{d}.')
            else:
                horizontal_shift = 0.
                vertical_shift = 0.

            # define new box with vertical orientation such that labelled part in center
            box = (step + horizontal_shift * d, mid_height - np.floor(d / 2) + vertical_shift * d,
                   step + (1 + horizontal_shift) * d, mid_height + np.ceil(d / 2) + vertical_shift * d)
            mask_cropped = mask_processed.crop(box)
            if mask_cropped.getbbox():  # save processed images if still some labelled part in mask_cropped
                filename_mask = os.path.join(dir_out, f'{name}_{idx:02d}_masked.png')
                mask_cropped.save(filename_mask)
                filename_img = os.path.join(dir_out, f'{name}_{idx:02d}.png')
                img_processed.crop(box).save(filename_img)
            else:
                logging.info('  Tile does not contain any labelled part after the processing. Skip this tile.')


path = os.path.abspath(os.pardir)

img_folder = 'dev'
tile_size = 512

path_img = os.path.join(path, 'data', 'raw', img_folder)
path_out = os.path.join(path, 'data', 'interim', f'{img_folder}_augmented_{tile_size}')
os.makedirs(path_out, exist_ok=True)

logging_config.define_root_logger(os.path.join(path_out, f'log.txt'))

for file in os.listdir(path_img):
    file_name, _ = os.path.splitext(file)
    if not file_name.endswith('_masked'):
        # augmented
        tile(file, path_img, path_out, tile_size,
             overlap=0.5,
             rot=[-90, 90],
             contrast=[0.8, 1.2],
             brightness=[0.8, 1.2],
             blur=[0., 1.],
             shift=[-0.3, 0.3])

        # alittleaugmented
        # tile(file, path_img, path_out, tile_size,
        #      overlap=0.5,
        #      rot=[-30, 30],
        #      contrast=[0.9, 1.1],
        #      brightness=[0.9, 1.1],
        #      blur=None,
        #      shift=[-0.15, 0.15])

        # notaugmented
        # tile(file, path_img, path_out, tile_size)
