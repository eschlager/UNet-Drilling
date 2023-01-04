# -*- coding: utf-8 -*-
"""
Created on 01.09.2022
@author: eschlager

Evaluation of multiple models from cross validation on tiles of evaluation set;
used after training in train_unet.py

"""

import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

import argparse
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle

matplotlib.use('Agg')

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.sep.join([script_dir, '..', 'src']))
import utils
from data_loader import DataLoader

home_dir = os.path.abspath(os.path.join(script_dir, '..'))


def main(gpu_id, model_path, data_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # directory of models to evaluate
    model_dir = os.path.join(home_dir, model_path)
    print(f"Make evaluation for model: {model_dir}")
    models_nr = len([name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name)) if
                     name.startswith('trained_model_fold')])
    print(f"  evaluate {models_nr} model folds...")

    # directory of data used for evaluation
    data_dir = os.path.join(home_dir, data_path)
    print(f"Make evaluation for data in : {data_dir}")

    # directory to save evaluation
    out_dir = os.path.join(home_dir, 'evaluation',
                           'model_' + os.path.basename(os.path.normpath(model_dir)),
                           'tiles_' + os.path.basename(os.path.normpath(data_dir)))
    os.makedirs(out_dir, exist_ok=True)

    # load configurations and data
    model_args = pickle.load(open(os.path.join(model_dir, "arguments.pkl"), 'rb'))
    tile_size = int(data_dir.split("_")[-1])
    wear_mode = model_args["wear_mode"]
    data = DataLoader(data_dir, tile_size, wear_mode)
    images, masks = data.load_imgs_and_masks()

    # load multi class masks
    if wear_mode == 3:
        masks_multi = masks.copy()
    else:
        data_multi = DataLoader(data_dir, tile_size, 3)
        _, masks_multi = data_multi.load_imgs_and_masks()

    num_classes = data.nr_classes
    scores = {'IOU': [], 'Dice': [], 'TPR': [], 'TNR': []}
    if num_classes > 1:
        scores_macro = deepcopy(scores)
        scores_micro = deepcopy(scores)
        scores_classes_joined = {'IOU': [], 'Dice': [], 'TPR': [], 'TNR': [], 'FNBGR_0': [], 'FNBGR_1': []}
        samples_avg = 'macro'
    else:
        samples_avg = ''
        scores = {'IOU': [], 'Dice': [], 'TPR': [], 'TNR': [], 'FNBGR_0': [], 'FNBGR_1': []}

    iou_samples = {}
    tpr_samples = {}
    tnr_samples = {}

    for ifold in range(0, models_nr):
        print(f"\nMake predictions for model fold {ifold}...")
        model_fold_dir = os.path.join(model_dir, f'trained_model_fold{ifold:02}.tf')

        model = tf.keras.models.load_model(model_fold_dir, compile=False)
        _, pred_cat = utils.predict_cat(model, images, batch_size=1)

        print(f"Calculate {samples_avg} scores per sample...")
        iou_samples[ifold] = utils.intersection_over_union(masks, pred_cat, average=samples_avg, reduce=False)
        tpr_samples[ifold] = utils.sensitivity(masks, pred_cat, average=samples_avg, reduce=False)
        tnr_samples[ifold] = utils.specificity(masks, pred_cat, average=samples_avg, reduce=False)

        print("Plot predicted masks...")
        for idx, (img, mask, pred) in enumerate(zip(images, masks, pred_cat)):
            fig = utils.plot_image_vs_mask_vs_pred(img, mask, pred, wear_mode)
            plt.savefig(os.path.join(out_dir, f'predicted_fold{ifold:02}_{idx:02}.jpg'), bbox_inches='tight')
            plt.close(fig)

        print("Calculate scores...")  # if multi class, this scores are per class
        scores['IOU'].append(utils.intersection_over_union(masks, pred_cat, average=None))
        scores['Dice'].append(utils.dice_coef(masks, pred_cat, average=None))
        scores['TPR'].append(utils.sensitivity(masks, pred_cat, average=None))
        scores['TNR'].append(utils.specificity(masks, pred_cat, average=None))

        print("Calculate FNBGR...")
        masks_0 = masks_multi[:, :, :, 1:2]
        masks_1 = masks_multi[:, :, :, 2:3]
        if num_classes > 1:
            pred_cat_bg = pred_cat[:, :, :, 0:1]
        else:
            pred_cat_bg = 1 - pred_cat
        fbgbr0 = (masks_0 * pred_cat_bg).sum(axis=(1, 2, 3)) / masks_0.sum(axis=(1, 2, 3))
        fbgbr1 = (masks_1 * pred_cat_bg).sum(axis=(1, 2, 3)) / masks_1.sum(axis=(1, 2, 3))

        if num_classes > 1:
            masks_nobg = 1 - masks[:, :, :, :1]
            pred_cat_nobg = 1 - pred_cat[:, :, :, :1]
            print("Calculate scores with all wear classes taken as one (e.g., transform to binary)...")
            print("shape masks: ", masks_nobg.shape)
            print("shape predictions: ", pred_cat_nobg.shape)
            scores_classes_joined['IOU'].append(
                utils.intersection_over_union(masks_nobg, pred_cat_nobg, average=None))
            scores_classes_joined['Dice'].append(utils.dice_coef(masks_nobg, pred_cat_nobg, average=None))
            scores_classes_joined['TPR'].append(utils.sensitivity(masks_nobg, pred_cat_nobg, average=None))
            scores_classes_joined['TNR'].append(utils.specificity(masks_nobg, pred_cat_nobg, average=None))
            scores_classes_joined['FNBGR_0'].append(np.nanmean(fbgbr0))
            scores_classes_joined['FNBGR_1'].append(np.nanmean(fbgbr1))

            print("Calculate macro average scores...")
            scores_macro['IOU'].append(utils.intersection_over_union(masks, pred_cat, average='macro'))
            scores_macro['Dice'].append(utils.dice_coef(masks, pred_cat, average='macro'))
            scores_macro['TPR'].append(utils.sensitivity(masks, pred_cat, average='macro'))
            scores_macro['TNR'].append(utils.specificity(masks, pred_cat, average='macro'))

            print("Calculate micro average scores...")
            scores_micro['IOU'].append(utils.intersection_over_union(masks, pred_cat, average='micro'))
            scores_micro['Dice'].append(utils.dice_coef(masks, pred_cat, average='micro'))
            scores_micro['TPR'].append(utils.sensitivity(masks, pred_cat, average='micro'))
            scores_micro['TNR'].append(utils.specificity(masks, pred_cat, average='micro'))
        else:
            scores['FNBGR_0'].append(np.nanmean(fbgbr0))
            scores['FNBGR_1'].append(np.nanmean(fbgbr1))

    print(f"Save scores to {out_dir}")
    with pd.ExcelWriter(os.path.join(out_dir, f"model_scores.xlsx")) as writer:
        if num_classes == 1:
            df_scores = pd.DataFrame(scores, dtype=float)
            if models_nr > 1:
                df_scores.loc['mean', :] = df_scores.mean(axis=0)
                df_scores.loc['std', :] = df_scores.std(axis=0)
            df_scores.to_excel(writer, sheet_name='scores', float_format="%.4f")
        else:
            df_classes_joined = pd.DataFrame(scores_classes_joined, dtype=float)
            if models_nr > 1:
                df_classes_joined.loc['mean', :] = df_classes_joined.mean(axis=0)
                df_classes_joined.loc['std', :] = df_classes_joined.std(axis=0)
            df_classes_joined.to_excel(writer, sheet_name='results_classes_joined', float_format="%.4f")

            df_macro = pd.DataFrame(scores_macro)
            if models_nr > 1:
                df_macro.loc['mean', :] = df_macro.mean(axis=0)
                df_macro.loc['std', :] = df_macro.std(axis=0)
            df_macro.to_excel(writer, sheet_name='results_macro', float_format="%.4f")

            df_micro = pd.DataFrame(scores_micro)
            if models_nr > 1:
                df_micro.loc['mean', :] = df_micro.mean(axis=0)
                df_micro.loc['std', :] = df_micro.std(axis=0)
            df_micro.to_excel(writer, sheet_name='results_micro', float_format="%.4f")

            df_class = pd.DataFrame()
            for key in scores.keys():
                df_key = pd.DataFrame(scores[key])
                df_key.columns = [key + '_' + str(nr) for nr in df_key.columns]
                df_class = df_class.join(df_key, how='outer')
            if models_nr > 1:
                df_class.loc['mean', :] = df_class.mean(axis=0)
                df_class.loc['std', :] = df_class.std(axis=0)
            df_class.to_excel(writer, sheet_name='results_class', float_format="%.4f")

    print("Plot Boxplots of (macro average if multiclass) score distributions...")

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.boxplot(iou_samples.values())
    plt.xticks(range(1, len(iou_samples.keys()) + 1), iou_samples.keys())
    ax.set_ylim([0, 1.01])
    plt.ylabel('IoU ' + samples_avg)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'boxplot_iou{samples_avg}.jpg'), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.boxplot(tpr_samples.values())
    plt.xticks(range(1, len(tpr_samples.keys()) + 1), tpr_samples.keys())
    ax.set_ylim([0, 1.01])
    plt.ylabel('Sensitivity ' + samples_avg)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'boxplot_sensitivity{samples_avg}.jpg'), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.boxplot(tnr_samples.values())
    plt.xticks(range(1, len(tnr_samples.keys()) + 1), tnr_samples.keys())
    ax.set_ylim([0, 1.01])
    plt.ylabel('Specificity ' + samples_avg)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'boxplot_specificity{samples_avg}.jpg'), bbox_inches='tight')
    plt.close()


parser = argparse.ArgumentParser(description='evaluate_model')
parser.add_argument('--gpu_id', default="1,2", type=str)
parser.add_argument('--model_folder', default='models/finalbn_alittleaug_iou_3_00', type=str,
                    help='Folder where models are saved.')
parser.add_argument('--data_folder', default='data/interim/dev_centertiles_512', type=str,
                    help='Folder of evaluation data.')
args = parser.parse_args()

if __name__ == "__main__":
    main(args.gpu_id, args.model_folder, args.data_folder)
