# -*- coding: utf-8 -*-
"""
Created on 18.08.2022
@author: eschlager

code from https://blog.paperspace.com/unet-architecture-image-segmentation/#tensorflow-implementation-of-u-net

"""

from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


def convolution_operation(entered_input, filters=64):
    conv1 = Conv2D(filters, kernel_size=(3, 3), padding="same")(entered_input)
    conv1 = BatchNormalization()(conv1)
    act1 = ReLU()(conv1)

    conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same")(act1)
    conv2 = BatchNormalization()(conv2)
    act2 = ReLU()(conv2)
    return act2


def encoder(entered_input, filters=64):
    enc1 = convolution_operation(entered_input, filters)
    MaxPool1 = MaxPooling2D(strides=(2, 2))(enc1)
    return enc1, MaxPool1


def decoder(entered_input, skip, filters=64):
    Upsample = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input)
    Connect_Skip = Concatenate()([Upsample, skip])
    out = convolution_operation(Connect_Skip, filters)
    return out


def U_Net(Image_Size, num_classes=1):
    input1 = Input(Image_Size, name='input_image')

    # Contracting path
    skip1, encoder_1 = encoder(input1, 64)
    skip2, encoder_2 = encoder(encoder_1, 64 * 2)
    skip3, encoder_3 = encoder(encoder_2, 64 * 4)
    skip4, encoder_4 = encoder(encoder_3, 64 * 8)

    conv_block = convolution_operation(encoder_4, 64 * 16)

    # Expanding path
    decoder_1 = decoder(conv_block, skip4, 64 * 8)
    decoder_2 = decoder(decoder_1, skip3, 64 * 4)
    decoder_3 = decoder(decoder_2, skip2, 64 * 2)
    decoder_4 = decoder(decoder_3, skip1, 64)

    if num_classes == 1:
        act_fun = "sigmoid"
    elif num_classes > 1:
        act_fun = "softmax"

    out = Conv2D(num_classes, 1, padding="same", activation=act_fun, name='output_image')(decoder_4)

    model = Model(input1, out)
    return model

# input_shape = (512, 512, 3)
# model = U_Net(input_shape)
# model.summary()
#
# plot_model(model, "model.png", show_shapes=False, show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)
