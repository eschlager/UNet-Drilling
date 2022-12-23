# -*- coding: utf-8 -*-
"""
Created on 01.09.2022
@author: eschlager

"""

import os
import sys

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib

import argparse
import pandas as pd
from copy import deepcopy
import pickle

matplotlib.use('Agg')

script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.sep.join([script_dir, '..']))
from common_data_paths import get_base_path_for_current_host
import utils
from data_loader import DataLoader


def main(gpu_id, model_folder, data_folder):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    path = get_base_path_for_current_host()

    model_path = os.sep.join([path, model_folder])
    print(f"Make evaluation for model: {model_path}")
    models_nr = len([name for name in os.listdir(model_path) if os.path.isdir(os.sep.join([model_path, name])) if
                     name.startswith('trained_model_fold')])
    print(f"  evaluate {models_nr} model folds...")

    data_path = os.sep.join([path, data_folder])
    print(f"Make evaluation for data in : {data_path}")
    tile_size = int(data_folder.split("_")[-1])

    model_args = pickle.load(open(os.path.join(model_path, "arguments.pkl"), 'rb'))
    wear_mode = model_args["wear_mode"]
    data = DataLoader(data_path, tile_size, wear_mode)
    images, masks = data.load_imgs_and_masks()

    print(masks.shape)

    eval_path = os.sep.join([model_path, 'evaluation_' + os.path.basename(os.path.normpath(data_folder))])
    os.makedirs(eval_path, exist_ok=True)

    num_classes = data.nr_classes
    scores = {'IOU': [], 'Dice': [], 'TPR': [], 'TNR': []}
    if num_classes > 1:
        scores_macro = deepcopy(scores)
        scores_micro = deepcopy(scores)
        scores_classes_micro = deepcopy(scores)
        samples_avg = 'macro'
    else:
        samples_avg = ''

    iou_samples = {}
    tpr_samples = {}
    tnr_samples = {}

    for ifold in range(0, models_nr):
        print(f"\nMake predictions for model fold {ifold}...")
        model_dir = os.path.sep.join([model_path, f'trained_model_fold{ifold:02}.tf'])

        model = tf.keras.models.load_model(model_dir, compile=False)
        # use custom_objects={'loss': utils.loss_functions(args.loss)} or compile=False to load model with custom loss fct.
        _, pred_cat = utils.predict_cat(model, images, batch_size=1)

        print(f"Calculate {samples_avg} scores per sample...")
        iou_samples[ifold] = utils.intersection_over_union(masks, pred_cat, average=samples_avg, reduce=False)
        tpr_samples[ifold] = utils.sensitivity(masks, pred_cat, average=samples_avg, reduce=False)
        tnr_samples[ifold] = utils.specificity(masks, pred_cat, average=samples_avg, reduce=False)

        print("Plot predicted masks...")
        for idx, (img, mask, pred) in enumerate(zip(images, masks, pred_cat)):
            fig = utils.plot_image_vs_mask_vs_pred(img, mask, pred, wear_mode)
            plt.savefig(os.path.sep.join([eval_path, f'predicted_fold{ifold:02}_{idx:02}.jpg']), bbox_inches='tight')
            plt.close(fig)

        print("Calculate scores...")
        # Attention: tpr without any positives does return 0, etc.
        scores['IOU'].append(utils.intersection_over_union(masks, pred_cat, average=None))
        scores['Dice'].append(utils.dice_coef(masks, pred_cat, average=None))
        scores['TPR'].append(utils.sensitivity(masks, pred_cat, average=None))
        scores['TNR'].append(utils.specificity(masks, pred_cat, average=None))

        if num_classes > 1:
            masks_nobg = masks[:, :, :, 1:]
            pred_cat_nobg = pred_cat[:, :, :, 1:]
            print("Calculate micro average scores of the non-background classes only...")
            print("shape masks: ", masks_nobg.shape)
            print("shape predictions: ", pred_cat_nobg.shape)
            scores_classes_micro['IOU'].append(
                utils.intersection_over_union(masks_nobg, pred_cat_nobg, average='micro'))
            scores_classes_micro['Dice'].append(utils.dice_coef(masks_nobg, pred_cat_nobg, average='micro'))
            scores_classes_micro['TPR'].append(utils.sensitivity(masks_nobg, pred_cat_nobg, average='micro'))
            scores_classes_micro['TNR'].append(utils.specificity(masks_nobg, pred_cat_nobg, average='micro'))

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

    print(f"Save scores to {eval_path}")
    with pd.ExcelWriter(os.path.sep.join([eval_path, f"model_scores.xlsx"])) as writer:
        if num_classes == 1:
            df_scores = pd.DataFrame(scores, dtype=float)
            df_scores.loc['mean', :] = df_scores.mean(axis=0)
            df_scores.loc['std', :] = df_scores.std(axis=0)
            df_scores.to_excel(writer, sheet_name='scores', float_format="%.4f")
        else:
            df_classes_micro = pd.DataFrame(scores_classes_micro)
            df_classes_micro.loc['mean', :] = df_classes_micro.mean(axis=0)
            df_classes_micro.loc['std', :] = df_classes_micro.std(axis=0)
            df_classes_micro.to_excel(writer, sheet_name='results_classes_micro', float_format="%.4f")

            df_macro = pd.DataFrame(scores_macro)
            df_macro.loc['mean', :] = df_macro.mean(axis=0)
            df_macro.loc['std', :] = df_macro.std(axis=0)
            df_macro.to_excel(writer, sheet_name='results_macro', float_format="%.4f")

            df_micro = pd.DataFrame(scores_micro)
            df_micro.loc['mean', :] = df_micro.mean(axis=0)
            df_micro.loc['std', :] = df_micro.std(axis=0)
            df_micro.to_excel(writer, sheet_name='results_micro', float_format="%.4f")

            df_class = pd.DataFrame()
            for key in scores.keys():
                df_key = pd.DataFrame(scores[key])
                df_key.columns = [key + '_' + str(nr) for nr in df_key.columns]
                df_class = df_class.join(df_key, how='outer')
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
    plt.savefig(os.path.sep.join([eval_path, f'boxplot_iou{samples_avg}.jpg']), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.boxplot(tpr_samples.values())
    plt.xticks(range(1, len(tpr_samples.keys()) + 1), tpr_samples.keys())
    ax.set_ylim([0, 1.01])
    plt.ylabel('Sensitivity ' + samples_avg)
    plt.tight_layout()
    plt.savefig(os.path.sep.join([eval_path, f'boxplot_sensitivity{samples_avg}.jpg']), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.boxplot(tnr_samples.values())
    plt.xticks(range(1, len(tnr_samples.keys()) + 1), tnr_samples.keys())
    ax.set_ylim([0, 1.01])
    plt.ylabel('Specificity ' + samples_avg)
    plt.tight_layout()
    plt.savefig(os.path.sep.join([eval_path, f'boxplot_specificity{samples_avg}.jpg']), bbox_inches='tight')
    plt.close()


parser = argparse.ArgumentParser(description='evaluate_model')
parser.add_argument('--gpu_id', default="0", type=str)
parser.add_argument('--model_folder', default='output/image_segmentation/unet_aug_ce_3_00', type=str,
                    help='Folder where models are saved.')
parser.add_argument('--data_folder', default='data/images/interim/dev_centertiles_512', type=str,
                    help='Folder of evaluation data.')
args = parser.parse_args()

if __name__ == "__main__":
    main(args.gpu_id, args.model_folder, args.data_folder)
