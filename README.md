# U-Net for Semantic Image Segmentation on Microscopic Drilling Tool Images for Wear Detection

This repo contains the code accompanying our paper [Evaluation of Data Augmentation and Loss Functions in Semantic Image Segmentation for Drilling Tool Wear Detection](https://arxiv.org/??????).


The microscopic images of cutting inserts of drilling tools have dimensions reaching from 4750 x 1200 pixels where 1 pixel equals 1.493μm, up to 11500 x 1500 pixels where 1 pixel equals 0.781μm. We distinguish two types of wear: abrasive wear, and build-up-edge, which are labeled in different colours.


To process this high resolution images, the overlap-tile strategy is applied as proposed in [Ronneberger et. al: U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1007/978-3-319-24574-4_28). Thus, for training and evaluation, the images are cut into a large number of smaller tiles which can easily be easily processed. The preprocessing pipeline can apply several augmentation techniques of varying intensity.

<img src="https://github.com/eschlager/UNet-Drilling/blob/master/dissemination/figures/flowchart_augmentation.jpg" height="200"> 


Based on the two different wear types, the code can be used in 4 different modes: 
* mode 0: build-up-edge only (binary problem)
* mode 1: abrasive wear only (binary problem)
* mode 2: build-up-edge and abrasive wear as one class (binary problem)
* mode 3: build-up-edge and abrasive wear as distinct classes (multi class problem)


Furthermore, three different types of loss functions are used:
* Cross Entropy
* Focal Crossentropy
* IoU-based loss



### Citation 
If you find this code useful, please consider citing our work:
```
@article{UNetDrilling,
  title={Evaluation of Data Augmentation and Loss Functions in Semantic Image Segmentation for Drilling Tool Wear Detection},
  author={Elke Schlager AND Andreas Windisch},
  journal={arXiv preprint arXiv:????.????},
  year={2023}
}
```
