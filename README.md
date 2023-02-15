# U-Net for Semantic Image Segmentation on Microscopic Drilling Tool Images for Wear Detection

This repo contains the code accompanying our paper [Evaluation of Data Augmentation and Loss Functions in Semantic Image Segmentation for Drilling Tool Wear Detection](https://arxiv.org/abs/2302.05262).

## Data and Data preparation
The microscopic images of cutting inserts of drilling tools have dimensions reaching from 4750 x 1200 pixels, up to 11500 x 1500 pixels. To process such high resolution images in U-Net, they are partitioned into smaller tiles. 
For training, the script `prepare/create_augmented_tiles.py` cuts smaller (overlapping) tiles from the whole images, and applies several augmentation techniques:

<img src="https://github.com/eschlager/UNet-Drilling/blob/main/figures/flowchart_augmentation.jpg" height="350"> 

We distinguish two types of wear: abrasive wear, coloured in blue, and build-up-edge, coloured in yellow. Routines in `src/data_loader.py` and `src/utils.py` are specific to these colour labels and have to be adjusted when applied to different coloured masks.


<img src="https://github.com/eschlager/UNet-Drilling/blob/main/figures/SONT_072907_ER_M30_CTPP430_90Bo_A.jpg" height="120">
<img src="https://github.com/eschlager/UNet-Drilling/blob/main/figures/SONT_072907_ER_M30_CTPP430_90Bo_A_masked.jpg" height="120"> 


## Training

Training (with cross validation) is performed in `train_unet.py`.


Based on the two different wear types, the code can be used in 4 different modes: 
* mode 0: build-up-edge only (binary problem)
* mode 1: abrasive wear only (binary problem)
* mode 2: build-up-edge and abrasive wear as one class (binary problem)
* mode 3: build-up-edge and abrasive wear as distinct classes (multi class problem)


Furthermore, three different types of loss functions are used:
* Cross Entropy
* Focal Crossentropy
* IoU-based loss

At the end of the training, the evaluation script is started, evaluating the trained models on an unseen development set, which has to be prepared using `prepare/create_augmented_tiles.py`.



## Predicting 

For predicting whole images, the overlap-tile strategy is applied as proposed in [Ronneberger et. al: U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28). 
The pipeline in `predictor.py` can be run in mode 0, which stores the predicted masks only, or in mode 1, which adds the predicted mask to the original image.
