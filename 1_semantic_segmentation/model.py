import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from metrics import *
from metrics_v2 import *
from miou import MeanIoU


def conv_block(inputs, filters, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal'):
    """Defines a convolutional block with two convolutional layers."""
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inputs)
    x = Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(x)
    return x

def upsample_and_concat(inputs, skip_connection, filters, kernel_size=2, activation='relu', padding='same', kernel_initializer='he_normal'):
    """Upsamples the input and concatenates with the skip connection."""
    up = Conv2D(filters, kernel_size, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(
        UpSampling2D(size=(2, 2))(inputs)
    )
    return concatenate([skip_connection, up], axis=3)

def unet(pretrained_weights=None, input_size=(256, 256, 1), dropout_rate=0.5, learning_rate=1e-4):
    inputs = Input(input_size)

    # Encoding path
    conv1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_block(pool1, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_block(pool2, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_block(pool3, 512)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # Bottleneck
    conv5 = conv_block(pool4, 1024)
    drop5 = Dropout(dropout_rate)(conv5)

    # Decoding path
    conv6 = upsample_and_concat(drop5, drop4, 512)
    conv6 = conv_block(conv6, 512)

    conv7 = upsample_and_concat(conv6, conv3, 256)
    conv7 = conv_block(conv7, 256)

    conv8 = upsample_and_concat(conv7, conv2, 128)
    conv8 = conv_block(conv8, 128)

    conv9 = upsample_and_concat(conv8, conv1, 64)
    conv9 = conv_block(conv9, 64)

    # Output layer
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)

    # Compile model
    miou_metric = MeanIoU(2)
    metrics = ['accuracy', miou_metric.mean_iou, iou, dice_coef]
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=metrics)

    # Load pretrained weights if specified
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
